"""bench_single_eval.py — Benchmark single-eval MoE computation vs per-layer eval.

This experiment tests whether building the entire 60-layer MoE expert computation
graph lazily and evaluating with a single mx.eval() can eliminate the ~1ms per-layer
eval overhead that dominates inference time.

The key idea: in two-pass mode, Pass 1 pre-computes routing indices and batch-reads
expert weights. Pass 2 currently re-runs routing per layer (mx.eval(inds) per layer).
This experiment skips the per-layer re-routing and uses Pass 1's routing directly,
building the entire computation graph lazily.

Approaches tested:
  A) Per-layer eval (current behavior in batch-experts mode: eval every 4 layers)
  B) Single eval at end (fully lazy graph, pre-computed routing)
  C) Separate Metal stream for expert compute, routing evals on default stream
  D) Per-layer eval with simulated routing evals (current two-pass Pass 2 behavior)

Uses real gather_qmm calls with 4-bit quantized weights, matching the actual
Qwen3.5 MoE computation pattern.
"""

import argparse
import time
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn


def compute_moe_layer(x, idx, gate_w, gate_s, gate_b, up_w, up_s, up_b,
                      down_w, down_s, down_b, group_size=64, bits=4):
    """One layer of MoE expert computation: gate+up -> SwiGLU -> down."""
    x_in = mx.expand_dims(x, (-2, -3))
    x_gate = mx.gather_qmm(x_in, gate_w, gate_s, gate_b, rhs_indices=idx,
                            transpose=True, group_size=group_size, bits=bits, mode='affine')
    x_up = mx.gather_qmm(x_in, up_w, up_s, up_b, rhs_indices=idx,
                          transpose=True, group_size=group_size, bits=bits, mode='affine')
    x_act = nn.silu(x_gate) * x_up
    out = mx.gather_qmm(x_act, down_w, down_s, down_b, rhs_indices=idx,
                         transpose=True, group_size=group_size, bits=bits, mode='affine')
    return out.squeeze(-2)


def make_expert_weights(n_layers, top_k, hidden, intermediate, group_size=64, bits=4):
    """Create 4-bit quantized expert weight tensors for all layers."""
    packed_dim = hidden // (32 // bits)  # 4-bit: hidden // 8
    scales_dim = hidden // group_size
    packed_dim_down = intermediate // (32 // bits)
    scales_dim_down = intermediate // group_size

    all_weights = {}
    for i in range(n_layers):
        all_weights[i] = {
            'gate_w': mx.random.randint(0, 255, shape=(top_k, intermediate, packed_dim)).astype(mx.uint32),
            'gate_s': mx.random.normal((top_k, intermediate, scales_dim)).astype(mx.bfloat16),
            'gate_b': mx.random.normal((top_k, intermediate, scales_dim)).astype(mx.bfloat16),
            'up_w': mx.random.randint(0, 255, shape=(top_k, intermediate, packed_dim)).astype(mx.uint32),
            'up_s': mx.random.normal((top_k, intermediate, scales_dim)).astype(mx.bfloat16),
            'up_b': mx.random.normal((top_k, intermediate, scales_dim)).astype(mx.bfloat16),
            'down_w': mx.random.randint(0, 255, shape=(top_k, hidden, packed_dim_down)).astype(mx.uint32),
            'down_s': mx.random.normal((top_k, hidden, scales_dim_down)).astype(mx.bfloat16),
            'down_b': mx.random.normal((top_k, hidden, scales_dim_down)).astype(mx.bfloat16),
        }
    # Materialize all
    all_arrays = []
    for i in range(n_layers):
        all_arrays.extend(all_weights[i].values())
    mx.eval(*all_arrays)
    return all_weights


def run_benchmark(n_layers, top_k, hidden, intermediate, n_warmup=3, n_runs=5):
    """Run all benchmark approaches and report results."""
    print(f"Configuration: {n_layers} layers, hidden={hidden}, intermediate={intermediate}, top_k={top_k}")
    print(f"Warmup: {n_warmup} runs, Measure: {n_runs} runs")
    print()

    # Create weights
    print("Creating weight tensors...", flush=True)
    weights = make_expert_weights(n_layers, top_k, hidden, intermediate)
    total_bytes = sum(sum(a.nbytes for a in weights[i].values()) for i in range(n_layers))
    print(f"Weight memory: {total_bytes / 1e6:.1f} MB")

    # Create input
    h_init = mx.random.normal((1, 1, hidden))
    mx.eval(h_init)

    # Pre-computed routing indices (simulating Pass 1 output)
    all_inds = [mx.array(np.arange(top_k))[None, None, :] for _ in range(n_layers)]
    mx.eval(*all_inds)

    # Simulated routing scores (uniform for this benchmark)
    all_scores = [mx.ones((1, 1, top_k)) / top_k for _ in range(n_layers)]
    mx.eval(*all_scores)

    results = {}

    # ---- APPROACH A: Eval every 4 layers (current batch-experts behavior) ----
    times_a = []
    for run in range(n_warmup + n_runs):
        h = h_init
        t0 = time.perf_counter()
        for i in range(n_layers):
            w = weights[i]
            y = compute_moe_layer(h, all_inds[i],
                                  w['gate_w'], w['gate_s'], w['gate_b'],
                                  w['up_w'], w['up_s'], w['up_b'],
                                  w['down_w'], w['down_s'], w['down_b'])
            y = (y * all_scores[i][..., None]).sum(axis=-2)
            h = h + y
            if i % 4 == 3 or i == n_layers - 1:
                mx.eval(h)
        t = time.perf_counter() - t0
        if run >= n_warmup:
            times_a.append(t)
    results['A_eval_every4'] = times_a
    h_ref = h  # save for correctness check

    # ---- APPROACH B: Single eval at end (fully lazy) ----
    times_b = []
    for run in range(n_warmup + n_runs):
        h = h_init
        t0 = time.perf_counter()
        for i in range(n_layers):
            w = weights[i]
            y = compute_moe_layer(h, all_inds[i],
                                  w['gate_w'], w['gate_s'], w['gate_b'],
                                  w['up_w'], w['up_s'], w['up_b'],
                                  w['down_w'], w['down_s'], w['down_b'])
            y = (y * all_scores[i][..., None]).sum(axis=-2)
            h = h + y
        mx.eval(h)
        t = time.perf_counter() - t0
        if run >= n_warmup:
            times_b.append(t)
    results['B_single_eval'] = times_b

    # ---- APPROACH C: Separate stream + routing evals on default ----
    s2 = mx.new_stream(mx.default_device())
    times_c = []
    for run in range(n_warmup + n_runs):
        h = h_init
        t0 = time.perf_counter()
        for i in range(n_layers):
            # Simulate per-layer routing eval on default stream
            fake_routing = mx.ones((1,))
            mx.eval(fake_routing)

            with mx.stream(s2):
                w = weights[i]
                y = compute_moe_layer(h, all_inds[i],
                                      w['gate_w'], w['gate_s'], w['gate_b'],
                                      w['up_w'], w['up_s'], w['up_b'],
                                      w['down_w'], w['down_s'], w['down_b'])
                y = (y * all_scores[i][..., None]).sum(axis=-2)
                h = h + y
        mx.eval(h)
        t = time.perf_counter() - t0
        if run >= n_warmup:
            times_c.append(t)
    results['C_separate_stream'] = times_c

    # ---- APPROACH D: Per-layer eval with routing evals (two-pass Pass 2 current) ----
    times_d = []
    for run in range(n_warmup + n_runs):
        h = h_init
        t0 = time.perf_counter()
        for i in range(n_layers):
            # Simulate attention
            h_mid = h + mx.random.normal((1, 1, hidden)) * 0.001

            # Simulate routing eval (this is the forced sync in current two-pass)
            gates = mx.random.normal((1, 1, 512))
            inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
            mx.eval(inds)  # forced sync for routing

            # Expert compute (uses pre-loaded weights, but routing forced sync above)
            w = weights[i]
            y = compute_moe_layer(h_mid, all_inds[i],
                                  w['gate_w'], w['gate_s'], w['gate_b'],
                                  w['up_w'], w['up_s'], w['up_b'],
                                  w['down_w'], w['down_s'], w['down_b'])
            y = (y * all_scores[i][..., None]).sum(axis=-2)
            h = h_mid + y
            # Two-pass defers eval to end, but routing eval forces partial sync
        mx.eval(h)
        t = time.perf_counter() - t0
        if run >= n_warmup:
            times_d.append(t)
    results['D_routing_evals'] = times_d

    # ---- APPROACH E: Per-layer eval (every single layer, worst case) ----
    times_e = []
    for run in range(n_warmup + n_runs):
        h = h_init
        t0 = time.perf_counter()
        for i in range(n_layers):
            w = weights[i]
            y = compute_moe_layer(h, all_inds[i],
                                  w['gate_w'], w['gate_s'], w['gate_b'],
                                  w['up_w'], w['up_s'], w['up_b'],
                                  w['down_w'], w['down_s'], w['down_b'])
            y = (y * all_scores[i][..., None]).sum(axis=-2)
            h = h + y
            mx.eval(h)
        t = time.perf_counter() - t0
        if run >= n_warmup:
            times_e.append(t)
    results['E_per_layer_eval'] = times_e

    # ---- APPROACH F: Routing + attention eval per-layer, expert compute fully lazy ----
    # This tests what happens when attention ops force partial graph evaluation
    # even though we try to defer expert compute. The key question: does
    # mx.eval(routing) also force evaluation of the expert graph from the
    # previous layer (since h feeds into the next layer's attention)?
    times_f = []
    for run in range(n_warmup + n_runs):
        h = h_init
        t0 = time.perf_counter()
        for i in range(n_layers):
            # Attention (depends on h which includes expert output from previous layer)
            h_mid = h + mx.random.normal((1, 1, hidden)) * 0.001

            # Route on h_mid (lazy, not eval'd yet)
            gates = mx.random.normal((1, 1, 512))  # simulated gate output
            inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]

            # ONLY eval the routing indices, not h_mid
            # BUT: does evaluating inds force h_mid (and thus previous layer's expert) to eval?
            # No! inds depends on gates (random), not on h. So this should be cheap.
            mx.eval(inds)

            # Expert compute stays lazy
            w = weights[i]
            y = compute_moe_layer(h_mid, all_inds[i],
                                  w['gate_w'], w['gate_s'], w['gate_b'],
                                  w['up_w'], w['up_s'], w['up_b'],
                                  w['down_w'], w['down_s'], w['down_b'])
            y = (y * all_scores[i][..., None]).sum(axis=-2)
            h = h_mid + y
        mx.eval(h)
        t = time.perf_counter() - t0
        if run >= n_warmup:
            times_f.append(t)
    results['F_routing_eval_expert_lazy'] = times_f

    # ---- APPROACH G: Realistic routing that DEPENDS on h (forced dependency) ----
    # This is the actual pattern: routing depends on h_post = layernorm(h_mid),
    # and h_mid = h + attention_output, where h includes previous layer's expert output.
    # So mx.eval(inds) MUST force evaluation of ALL prior expert computation.
    times_g = []
    for run in range(n_warmup + n_runs):
        h = h_init
        t0 = time.perf_counter()
        for i in range(n_layers):
            # h_post depends on h (which includes expert output from all previous layers)
            h_post = h * 1.001  # simulated layernorm(h_mid)

            # Route based on h_post — DEPENDENCY on h!
            gates = mx.broadcast_to(mx.sum(h_post, axis=-1, keepdims=True), (1, 1, 512))
            inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]

            # This eval forces ALL prior computation because inds depends on h
            mx.eval(inds)

            # Expert compute
            w = weights[i]
            y = compute_moe_layer(h, all_inds[i],
                                  w['gate_w'], w['gate_s'], w['gate_b'],
                                  w['up_w'], w['up_s'], w['up_b'],
                                  w['down_w'], w['down_s'], w['down_b'])
            y = (y * all_scores[i][..., None]).sum(axis=-2)
            h = h + y
        mx.eval(h)
        t = time.perf_counter() - t0
        if run >= n_warmup:
            times_g.append(t)
    results['G_dependent_routing_eval'] = times_g

    # ---- Print results ----
    print()
    print("=" * 80)
    print(f"RESULTS ({n_layers} layers, hidden={hidden}, intermediate={intermediate}, top_k={top_k})")
    print("=" * 80)
    print(f"{'Approach':<40} {'Mean':>8} {'Min':>8} {'Max':>8} {'Speedup':>8}")
    print("-" * 80)

    baseline_mean = np.mean(results['E_per_layer_eval']) * 1000

    for name, times in sorted(results.items()):
        mean_ms = np.mean(times) * 1000
        min_ms = np.min(times) * 1000
        max_ms = np.max(times) * 1000
        speedup = baseline_mean / mean_ms
        print(f"{name:<40} {mean_ms:>7.1f}ms {min_ms:>7.1f}ms {max_ms:>7.1f}ms {speedup:>7.2f}x")

    print("-" * 80)
    print(f"Eval overhead per layer (A vs B): "
          f"{(np.mean(results['A_eval_every4']) - np.mean(results['B_single_eval'])) * 1000 / n_layers:.2f}ms")
    print(f"Per-layer eval overhead (E vs B): "
          f"{(np.mean(results['E_per_layer_eval']) - np.mean(results['B_single_eval'])) * 1000 / n_layers:.2f}ms")
    print(f"Routing-dependent eval penalty (G vs B): "
          f"{(np.mean(results['G_dependent_routing_eval']) - np.mean(results['B_single_eval'])) * 1000 / n_layers:.2f}ms")
    print()

    # Key insight
    b_mean = np.mean(results['B_single_eval']) * 1000
    e_mean = np.mean(results['E_per_layer_eval']) * 1000
    g_mean = np.mean(results['G_dependent_routing_eval']) * 1000
    print("KEY FINDINGS:")
    print(f"  Pure GPU compute (single eval, no sync overhead): {b_mean:.1f}ms")
    print(f"  Per-layer eval overhead: {e_mean - b_mean:.1f}ms ({(e_mean - b_mean) / n_layers:.2f}ms/layer)")
    print(f"  Dependent routing forces graph sync: {g_mean - b_mean:.1f}ms extra")
    print(f"  Single eval eliminates {(e_mean - b_mean) / e_mean * 100:.0f}% of total compute time")
    print()

    if g_mean > b_mean * 1.5:
        print("CONCLUSION: Routing that depends on h forces full graph evaluation per layer.")
        print("  The two-pass approach with re-routing in Pass 2 will NOT benefit from")
        print("  deferred eval because mx.eval(inds) implicitly evaluates the full chain.")
        print("  To get single-eval benefits, Pass 2 must use PASS 1's routing directly")
        print("  (no re-routing), accepting the ~10% approximation error.")
    else:
        print("CONCLUSION: Graph dependency overhead is manageable.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark single-eval MoE computation")
    parser.add_argument("--layers", type=int, default=60, help="Number of MoE layers")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--intermediate", type=int, default=1024, help="MoE intermediate dimension")
    parser.add_argument("--top-k", type=int, default=4, help="Active experts per token")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Measurement runs")
    args = parser.parse_args()

    run_benchmark(
        n_layers=args.layers,
        top_k=args.top_k,
        hidden=args.hidden,
        intermediate=args.intermediate,
        n_warmup=args.warmup,
        n_runs=args.runs,
    )


if __name__ == "__main__":
    main()

# ANE Research — Pushing the Edges of LLM Inference on Apple Silicon

## Mission

Find the absolute biggest Qwen3.5 variant that runs on an M3 Max (48GB) with usable inference performance. Qwen3.5-27B already works — anything below 35B is useless. We're targeting 35B+ models, potentially up to 397B, using every lever available: flash offloading, MoE sparsity, ANE co-processing, speculative decoding.

This follows the autoresearch paradigm: autonomous experimentation, measured results, iterative progress.

## Hardware

- **Chip**: Apple M3 Max (16-core CPU: 12P + 4E, 40-core GPU, 16-core ANE)
- **Memory**: 48 GB unified (~400 GB/s bandwidth, shared across CPU/GPU/ANE)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured), 335GB free
- **macOS**: 26.2 (Darwin 25.2.0)
- **ANE**: 16-core, ~16 TFLOPS FP16 (claimed ~6x inference perf — to validate)

## Target Models

| Model | Total Params | Active Params | 4-bit Size | Fits DRAM? |
|-------|-------------|---------------|------------|------------|
| Qwen3.5-35B-A3B | 35B | 3B | ~18GB | Yes |
| Qwen3.5-122B-A10B | 122B | 10B | ~61GB | No |
| Qwen3.5-397B-A17B | 397B | 17B | ~198GB | No |

mlx-community has pre-quantized 4-bit versions. We use MLX as our inference engine.

## The Levers

1. **Flash offloading** (LLM in a Flash) — stream weights from SSD instead of holding all in DRAM
2. **MoE sparsity** — only active experts needed per token (3-17B of 35-397B total)
3. **Selective persistence** — pin attention + routing in DRAM, stream expert FFN from flash
4. **Expert caching** — sliding window keeps recently-used experts in DRAM
5. **ANE co-processing** — offload attention to ANE while GPU handles expert FFN
6. **Speculative decoding** — draft with fast small model, verify with big model

## Safety Constraint

This is the primary machine. **NO experiments that risk OOM or system instability.** We stream weights explicitly, control memory usage, and release pages after use. Never hold more than ~20-25GB of model weights in DRAM at once.

## Autoresearch Protocol

### The 5-Minute Rule (HARD CONSTRAINT)

Every experiment gets a **5-minute wall-clock budget**. Non-negotiable:
- If not showing results within 5 minutes, kill it and move on
- Timeout = crash, log it, revert, next idea

### The Loop

LOOP FOREVER:
1. **Hypothesis** — what to try and why
2. **Implement** — write the code change
3. **git commit** — snapshot before running
4. **Run** — execute with 5-minute timeout
5. **Read results** — extract metrics
6. **Decide** — keep (improved) / discard (regressed) / crash (failed)
7. **Log** — append to results.tsv
8. **GOTO 1** — never stop, never ask

### Metrics & results.tsv

```
commit	model	params_B	active_B	tok_sec	ttft_ms	mem_gb	status	description
```

**"Improved" means**: bigger model running, OR same model faster (headroom to go bigger). North star: maximum params_B at usable tok/sec.

### Rules
- **Never stop** — run until interrupted
- **Never ask** — no "should I continue?"
- **Simplicity wins** — ugly complexity for marginal gains isn't worth it
- **Safety first** — no OOM, no system instability
- **Crashes are data** — log them and move on

## Project Structure

```
CLAUDE.md                   # this file
pyproject.toml              # uv project (mlx, mlx-lm, safetensors, psutil)
bench.py                    # MLX inference benchmark (tok/sec, memory)
stream_infer.py             # streaming inference engine (layer-by-layer, controlled mem)
progress.py                 # visualization: results.tsv -> progress.png
run.sh                      # 5-minute timeout wrapper
results.tsv                 # experiment log (git-ignored)
documentation/
  autoresearch/             # karpathy's autoresearch reference
  ANE/                      # maderix's ANE reverse-engineering project
  2312.11514v3.pdf          # Apple's "LLM in a Flash" paper
```

## Key Technical Context

### LLM in a Flash (Apple, 2023)
- Store model on flash, load selectively to DRAM during inference
- **Windowing**: cache recently-activated neurons, load only incremental delta per token
- **Row-column bundling**: bundle FFN up-projection columns with down-projection rows for larger sequential reads
- **Selective persistence**: keep attention (~1/3 of model) always in DRAM
- **Sparsity**: ReLU FFN layers have 90-97% sparsity — only 3-10% of weights needed
- Result: models up to 2x DRAM size, 4-20x speedup over naive loading

### ANE Reference (maderix/ANE)
- Reverse-engineered `_ANEClient` / `_ANECompiler` private APIs
- Dynamic weight pipeline: 10 kernels compiled once, weights via IOSurface spatial packing
- GPU+ANE zero-copy via shared IOSurface (demonstrated for inference)
- Qwen3-0.6B training on ANE: 412 ms/step on M4
- ANE peak: ~16 TFLOPS FP16 on M3

### M3 Max Specific
- Same 16-core NE as M3 Pro (Max adds GPU cores, not NE cores)
- ANE dynamic pipeline uses matmul (not conv), so ch=512 constraint doesn't apply
- MIL `program(1.3)` / `ios18` format works
- 17.5 GB/s SSD read = 3x faster than M1 Max in the LLM-in-a-Flash paper

## Experimental Results

### Qwen3.5-35B-A3B (4-bit, ~19GB — fits in 48GB DRAM)

Architecture: 40 layers (30 linear attention/SSM + 10 full attention), 256 experts, 8 active per token, hidden=2048

| Mode | tok/s | Peak Mem | Notes |
|------|-------|----------|-------|
| baseline | 6.87 | 9.5 GB | mlx_lm native stream_generate |
| layerwise | 9.38 | 8.4 GB | manual forward, per-layer mx.eval() |
| stream | 3.42 | 13.2 GB | reload from safetensors per layer per token |
| lazy (20tok) | 5.34 | 18.3 GB | mmap, OS pages on demand |
| lazy (50tok) | 10.84 | 18.2 GB | ~43 tok/s generation after warmup |

Key findings:
- **SSD throughput**: 474MB/layer cold load in ~22ms = **~20 GB/s** (matches hardware spec)
- **Warm (page-cached)**: ~1.5ms per layer (essentially free)
- **Per-layer compute**: 0.54ms linear_attn, 0.51ms full_attn (generation phase)
- **Lazy mode is best**: lowest initial memory (0.9 GB), OS manages paging efficiently
- **Manual forward matches native exactly** (max logit diff = 0.0)

### Expert Routing Analysis (35B)
- Routing is **moderately diverse**: 43-57% of 256 experts activated across 30 tokens
- Consecutive token overlap: 8-34% per layer (not very sticky)
- LRU cache hit rates: 16-expert=49%, 32=59%, 64=68%, 128=71%
- Implication: expert caching helps but doesn't eliminate SSD reads

### 122B Memory Architecture
- 48 layers (36 linear + 12 full), 256 experts, 8 active
- **95% of model weight is MoE experts** (58GB of 61GB)
- Active expert data per token: 1.81GB (out of 58GB)
- Non-expert weights (attention, embed, norm, gate, lm_head): ~3GB
- Strategy: pin 3GB non-expert, OS page cache handles expert paging

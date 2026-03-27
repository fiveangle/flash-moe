# Running Flash-MoE

This guide covers how to run inference using the `streamchat` CLI wrapper.

## Prerequisites

1. Download the model from HuggingFace:
   ```bash
   huggingface-cli download mlx-community/Qwen3.5-397B-A17B-4bit
   ```

2. Build the binaries:
   ```bash
   cd metal_infer && make
   ```

3. Create Python virtual environment (streamchat auto-detects and uses it):
   ```bash
   cd metal_infer
   python3 -m venv .venv
   .venv/bin/pip install numpy
   ```

## Quick Start

```bash
./streamchat "Hello world"
```

On first run, streamchat will automatically:
1. Extract non-expert weights (~5.5GB) to `metal_infer/model_weights.bin`
2. Create vocabulary binary from `tokenizer.json` to `metal_infer/vocab.bin`
3. If 4-bit expert weights are missing, prompt to choose extraction mode
4. Prompt to save settings to config file
5. Begin expert weight extraction (30-60 minutes for 4-bit)
6. Run inference

## Running streamchat

```bash
./streamchat "Your prompt here"
./streamchat --2bit "Your prompt"         # use 2-bit quantized experts
./streamchat "Hello" --tokens 50          # with extra options
./streamchat "Hello" --tokens 20 --timing  # with timing breakdown
```

## Expert Weight Extraction

### When Extraction is Required

The extraction dialog appears when:
- 4-bit expert weights are not present
- You pass `--2bit` but 2-bit expert weights are not present (and 4-bit exists)

If expert weights already exist, the dialog is skipped and inference runs immediately.

### Extraction Options

| Option | Size | Quality | Notes |
|--------|------|---------|-------|
| `4` - 4-bit mode | ~218GB | Full | Tool calling supported |
| `2` - 2-bit mode | ~120GB | Reduced | Faster, breaks JSON/tool calling |
| `Q` - Quit | - | - | Run later with `./streamchat` |

After selecting the quantization mode, you'll be asked if you want to save your preferences to a config file.

### 2-bit Mode

To use 2-bit quantized experts (faster but breaks tool calling):

```bash
./streamchat --2bit "Your prompt"
```

If 4-bit experts exist but 2-bit don't, streamchat will prompt you to extract 2-bit weights.

## Configuration File

streamchat supports a config file for default settings:
- `~/.streamchatrc` (user-wide)
- `./streamchatrc` (project-local, takes precedence)

Config is created automatically on first run if you choose to save settings.

```bash
# Example ~/.streamchatrc
DEFAULT_QUANT="4bit"
DEFAULT_TOKENS="20"
DEFAULT_TIMING="no"
```

## Setup Artifacts

| File | Size | Description |
|------|------|-------------|
| `metal_infer/model_weights.bin` | 5.5GB | Non-expert weights (mmap'd) |
| `metal_infer/model_weights.json` | 371KB | Manifest for weight loading |
| `metal_infer/vocab.bin` | 7.8MB | Tokenizer vocabulary |
| `<model>/packed_experts/` | 218GB | 4-bit expert weights |
| `<model>/packed_experts_2bit/` | 120GB | 2-bit expert weights |

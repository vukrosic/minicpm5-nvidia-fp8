# MiniCPM5-1B FP8 for NVIDIA GPUs

An optimized default for running
[`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) as one
active request on an NVIDIA Ampere-or-newer GPU with sufficient VRAM.

**RTX 3060 benchmark: 248.52 decode tok/s — 1.503× the matched BF16 vLLM
reference.**

![MiniCPM5-1B RTX 3060 speed and quality benchmark](docs/assets/benchmark-summary.svg)

## Benchmarks

### Speed

| Metric | BF16 reference | **FP8 default** | Change |
| --- | ---: | ---: | ---: |
| Median decode throughput | 165.39 tok/s | **248.52 tok/s** | **+50.3%** |
| Median end-to-end throughput | 152.42 tok/s | **221.42 tok/s** | **+45.3%** |
| Median TTFT | 38.60 ms | **35.11 ms** | **−9.0%** |

Speed used six prompts, one warmup and five timed repetitions per prompt. Batch
size and concurrency were one.

### Quality

| Benchmark | BF16 reference | **FP8 default** | Delta |
| --- | ---: | ---: | ---: |
| HumanEval+ | 95/164 | **96/164** | +1 |
| GSM8K slice | 19/50 | **18/50** | −1 |
| MMLU slice | 16/50 | **18/50** | +2 |
| C-Eval slice | 16/50 | **18/50** | +2 |
| IFEval slice | 38/50 | **33/50** | −5 |
| Four-slice total | 89/200 | **87/200** | −2 |

The quality runs were deterministic. FP8 changes outputs: HumanEval+ contained
eight losses and nine gains relative to BF16; the four-slice suite contained
12 losses and 10 gains. The task-level receipts are published under
[`results/`](results/).

## What was optimized

The default profile combines:

1. vLLM compiled execution and CUDA graph replay;
2. online per-block FP8 quantization with Marlin linear kernels;
3. Triton attention, the strongest measured decode backend on the RTX 3060;
4. a batch-one scheduler with one maximum active sequence;
5. prefix caching disabled for the single-request target.

The **combined profile** produced the measured speedup. The repository does not
claim a separate percentage for each component.

## Run it

The measured environment used Linux, an RTX 3060 12 GB, vLLM `0.27.1`, PyTorch
`2.13.0+cu130`, and model revision
`4e9de7a0778dc1c362e983e6858f0e77542cbdca`.

Download the official checkpoint directly on the GPU machine:

```bash
hf download openbmb/MiniCPM5-1B \
  --revision 4e9de7a0778dc1c362e983e6858f0e77542cbdca \
  --local-dir /workspace/models/MiniCPM5-1B
```

Install and launch. FP8 is selected automatically—no profile flag is needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt

export MINICPM5_MODEL_PATH=/workspace/models/MiniCPM5-1B
./serve --host 127.0.0.1 --port 8000
```

Inspect the exact vLLM command without starting the server:

```bash
./serve --model "$MINICPM5_MODEL_PATH" --dry-run
```

The launcher checks the selected NVIDIA GPU for compute capability 8.0 or
newer and never downloads weights. It labels the RTX 3060 as measured and
other accepted GPU architectures as compatible but unbenchmarked. Available
VRAM and the installed vLLM wheel still determine whether startup succeeds.

## GPU compatibility

| Hardware | Default profile |
| --- | --- |
| NVIDIA Ampere or newer (SM80+) | Architecture-compatible; benchmark separately |
| RTX 3060 12 GB | Supported and benchmarked here |
| NVIDIA Turing (SM75) | Not supported by this BF16 profile |
| AMD, Intel, Apple GPU or CPU | Not supported by this package |

vLLM can use Marlin FP8 weight-only kernels on older NVIDIA hardware, but this
release also uses BF16, which makes SM80+ the effective floor. Turing would
need a separately tested FP16 profile. See the
[vLLM FP8 documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/llm_compressor/fp8.md).

## What this release contains

- the default NVIDIA SM80+ FP8 serving profile and one-command launcher;
- the frozen benchmark specifications and document-hash manifest;
- full task-level paired comparison receipts;
- a reproducible chart generator and CPU-safe test suite.

The official OpenBMB weights are unchanged and are not redistributed. The BF16
configuration in this repository exists only to reproduce the benchmark
reference; it is not a second end-user release profile.

## Reproduce the evidence

Read [the full benchmark protocol](docs/BENCHMARKS.md), then inspect:

- [`release-summary.json`](results/release-summary.json);
- [`humanevalplus-comparison.json`](results/humanevalplus-comparison.json);
- [`release-intelligence-comparison.json`](results/release-intelligence-comparison.json);
- [`release-intelligence-manifest.json`](results/release-intelligence-manifest.json).

Regenerate the chart and run the local validation suite:

```bash
python3 scripts/render_benchmark_chart.py \
  --decision results/release-summary.json \
  --output docs/assets/benchmark-summary.svg

python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Scope

The software profile is released for NVIDIA SM80+ GPUs. The measured speedup
applies directly only to the pinned checkpoint, software, single-request
workload and tested RTX 3060. Other GPUs may select different vLLM kernels and
must be benchmarked separately. The four 50-task domain slices are release
screens, not full official leaderboard reproductions.

## Attribution and licenses

This project is independent research and is not an official OpenBMB release.
Repository code is MIT licensed. MiniCPM5-1B is published by OpenBMB under
Apache-2.0; obtain the model and its terms from the
[official model page](https://huggingface.co/openbmb/MiniCPM5-1B).

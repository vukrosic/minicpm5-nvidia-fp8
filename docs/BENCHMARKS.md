# Benchmark protocol and results

## Release question

> What is the fastest tested MiniCPM5-1B profile for one active request on one
> RTX 3060 12 GB, and what quality changes were observed against BF16?

The selected answer is online block FP8 at 248.52 decode tok/s, 1.503× BF16.
It is released as a speed-quality tradeoff because matched evaluations found
both losses and gains.

## Frozen environment

- Model: `openbmb/MiniCPM5-1B`
- Revision: `4e9de7a0778dc1c362e983e6858f0e77542cbdca`
- GPU: NVIDIA GeForce RTX 3060 12 GB
- Driver: `580.173.02`
- CUDA runtime: `13.0`
- vLLM: `0.27.1`
- PyTorch: `2.13.0+cu130`
- Batch size: 1
- Concurrent requests: 1
- Prefix caching: disabled
- Attention backend: Triton attention
- Decoding: greedy, temperature 0
- Thinking mode for quality evaluation: disabled

## Speed protocol

The speed suite used six fixed prompt classes, one warmup per prompt, five
timed repetitions per prompt, and 64 forced output tokens. That produced 30
timed cells per family. Decode throughput excludes TTFT; TTFT and end-to-end
throughput are recorded separately.

| Metric | BF16 | Online block FP8 | Candidate / BF16 |
| --- | ---: | ---: | ---: |
| Median decode throughput | 165.3899 tok/s | 248.5162 tok/s | 1.5026× |
| Median end-to-end throughput | 152.4231 tok/s | 221.4245 tok/s | 1.4527× |
| Median TTFT | 38.60 ms | 35.11 ms | 0.9096× |

Compilation and model load are not hidden inside warm decode throughput. One
fresh FP8 process on the already prepared host reached a healthy endpoint in
approximately 142 seconds. Installation, cache-cleared startup, and warm-start
timing were not measured.

## HumanEval+

The full 164-task suite used the same tasks, prompt construction, deterministic
decoding, extraction, and EvalPlus `0.3.1` executable checker for both families.

| Family | Passed | Paired losses | Paired gains | Exact token agreement |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 95/164 | Reference | Reference | Reference |
| Online block FP8 | 96/164 | 8 | 9 | 62/164 |

The aggregate gain of one task does not imply unchanged behavior. Task IDs for
all losses and gains are in
[`results/humanevalplus-comparison.json`](../results/humanevalplus-comparison.json).

## Frozen cross-domain suite

The suite was frozen before either family ran. Each selected document has a
dataset index and SHA-256 hash in the manifest.

| Domain | Selection | BF16 | FP8 | Losses / gains | Decision/output agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K | First 50 test items; five-shot | 19/50 | 18/50 | 5 / 4 | 27/50 |
| MMLU | Five items from each of ten frozen subjects | 16/50 | 18/50 | 0 / 2 | 48/50 |
| C-Eval | Five items from each of ten frozen subjects | 16/50 | 18/50 | 1 / 3 | 45/50 |
| IFEval | First 50 tasks; strict prompt accuracy | 38/50 | 33/50 | 6 / 1 | 2/50 |
| **Total** | **200 frozen items** | **89/200** | **87/200** | **12 / 10** | **122/200** |

Dataset revisions:

- GSM8K: `740312add88f781978c0658806c59bc2815b9866`
- MMLU: `c30699e8356da336a370243923dbaf21066bb9fe`
- C-Eval: `617524a00b307ff6f9933702f724131fe12ca7ce`
- IFEval: `966cd89545d6b6acfd7638bc708b98261ca58e84`

The exact paired-direction p-value over 22 discordant outcomes was `0.832`.
This does not prove equality. It says this finite subset did not establish a
directional imbalance between losses and gains. The deterministic scores will
repeat under the same software and prompts; uncertainty comes from which tasks
were selected and how well they represent general use.

The four evaluator calls took 246.0 seconds for BF16 and 208.4 seconds for FP8,
7.6 minutes combined, excluding model startup and compilation.

## Reproducing the cross-domain evaluator

Install the GPU evaluation dependencies from `requirements-gpu.txt`, start one
profile, and run each domain against the OpenAI-compatible completion endpoint:

```bash
for domain in gsm8k mmlu ceval ifeval; do
  python3 src/release_intelligence_suite.py run-domain \
    --spec benchmarks/specs/release-intelligence-mini-v1.json \
    --manifest results/release-intelligence-manifest.json \
    --domain "$domain" \
    --family bf16-vllm \
    --model-name openbmb/MiniCPM5-1B-bf16-rtx3060 \
    --base-url http://127.0.0.1:8000/v1/completions \
    --tokenizer "$MINICPM5_MODEL_PATH" \
    --output "runs/bf16/$domain.json"
done
```

Stop BF16, launch `recommended`, repeat with family
`online-block-fp8` and model name
`openbmb/MiniCPM5-1B-fp8-rtx3060`, then compare:

```bash
python3 src/release_intelligence_suite.py compare \
  --spec benchmarks/specs/release-intelligence-mini-v1.json \
  --reference-dir runs/bf16 \
  --candidate-dir runs/fp8 \
  --output runs/comparison.json
```

The runner refuses to overwrite receipts and validates every selected document
against the frozen manifest before accepting a result.

## Interpretation

The fixed-suite observation is an aggregate two-task regression, concentrated
most visibly in IFEval. General quality remains uncertain because these are
finite slices. The release claim is therefore limited to a measured 1.503×
single-request decode speedup with disclosed benchmark churn.

# MiniCPM5-1B FP8 is now the default

This release is the optimized RTX 3060 runtime—not a repackaging of the
original model.

## Measured result

- **248.52 decode tok/s** for one active request;
- **1.503×** the matched BF16 vLLM reference;
- **221.42 end-to-end output tok/s**;
- **35.11 ms** median TTFT.

## Optimized stack

- vLLM compiled execution and CUDA graph replay;
- online per-block FP8 with Marlin linear kernels;
- Triton attention;
- batch-one scheduling with one active sequence;
- prefix caching disabled for the single-request target.

## Published quality evidence

- HumanEval+: **96/164 FP8** vs **95/164 BF16**;
- frozen GSM8K/MMLU/C-Eval/IFEval slices: **87/200 FP8** vs **89/200 BF16**;
- every paired loss and gain is included in the repository.

FP8 is the default user path. The BF16 configuration remains only as a
benchmark reference. The official OpenBMB checkpoint is unchanged and must be
downloaded directly on the GPU machine.

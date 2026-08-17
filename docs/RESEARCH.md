# Research notes

## What produced the release speedup

The recommended profile combines several parts of vLLM's decode stack rather
than one isolated kernel:

1. compiled execution and CUDA graph replay reduce Python and launch overhead;
2. Triton attention was the strongest sustained-decode backend on this RTX
   3060 screen;
3. online block FP8 lets Marlin accelerate the linear layers without creating
   or redistributing a new checkpoint;
4. batch size and maximum concurrent sequences are fixed to one because the
   target is one active request, not server throughput.

The result is 248.52 decode tok/s versus 165.39 for the matched BF16 vLLM
reference.

## Why quality was evaluated separately

FP8 changes arithmetic and generated outputs. Token equality is therefore not
the appropriate product gate. We instead report deterministic task correctness,
every paired loss and gain, and exact output agreement as separate quantities.

The first small development canary showed no paired loss. Full HumanEval+ later
found eight losses and nine gains. The frozen cross-domain suite found 12 losses
and 10 gains. Keeping all three stages prevents the small screen from becoming
an overclaim.

## Corrected evaluator mistakes

Two failed integrations are intentionally documented:

- lm-eval's non-tokenized chat path could not score multiple-choice
  log-likelihood requests. The release runner uses tokenized requests and the
  official chat template with thinking disabled.
- the first comparator treated floating log-likelihood vectors as
  multiple-choice “output agreement.” Correctness was unaffected, but agreement
  was misleading. The canonical comparator records the selected answer choice.

The invalid draft comparison was not used for the release decision.

## Product decision

Online block FP8 is the recommended performance profile. BF16 remains the safe
fallback. The repository does not claim unchanged intelligence, mathematical
equivalence, or portability of the measured speedup to another GPU.

# Gemma4 HPU validation notes

This local branch records the current Gemma4 text-generation baseline on Intel Gaudi2 via vLLM Gaudi.

## Environment

- Hardware: Intel Gaudi2 (`HL-225`), 96 GiB HBM reported by `hl-smi`.
- HF cache: `/workspace/models/hub`.
- vLLM commit: `0a54df28471be07b3d668ea21c5e411569d3baea`.
- vLLM Gaudi branch baseline: Gemma4 HPU wrappers with head-size 512 paged attention support. Gemma4 prompt attention now defaults to HPU `fsdpa_impl` after short text correctness validation.

## Validated models

### `google/gemma-4-E2B-it`

Local snapshot:

```text
/workspace/models/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db
```

Naive baseline regression log:

```text
/tmp/gemma4-e2b-regression-final-20260527T094554Z.log
```

Observed deterministic outputs:

- `대한민국의 수도는 어디야?` -> `대한민국의 수도는 서울입니다.`
- `The model is running on Intel Gaudi.` translation -> `모델은 Intel Gaudi에서 실행 중입니다.`
- `3과 5를 더하면 얼마야?` -> `8`

### `RedHatAI/gemma-4-31B-it-FP8-block`

Local snapshot:

```text
/workspace/models/hub/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots/f676bf1357a9d27a77932dd4bf19d619724e74f6
```

Downloaded shard sizes were verified exactly:

```text
model-00001-of-00002.safetensors 26881224964 bytes
model-00002-of-00002.safetensors 6382013964 bytes
```

`safetensors.safe_open(..., framework="pt", device="cpu")` opened both shards successfully.

Naive baseline smoke log:

```text
/tmp/gemma4-31b-fp8-smoke-main-20260527T093748Z.log
```

Load/runtime evidence:

- vLLM resolved architecture: `Gemma4ForConditionalGeneration`.
- Quantization: `compressed-tensors`.
- HPU backend: `HPUAttentionV1`.
- Naive baseline prompt attention implementation: `naive_impl`.
- Model weights on HPU: `32.5980 GB`.
- KV cache allocation at `gpu_memory_utilization=0.80`: `49.63 GiB`.
- Total HPU memory used after cache init: `82.02 GiB / 94.62 GiB`.

Observed deterministic outputs:

- `대한민국의 수도는 어디야? 한 문장으로 답해줘.` -> `대한민국의 수도는 서울입니다.`
- `7 더하기 5는 얼마야? 숫자만 답해.` -> `12`

## Current scope and risks

This is a correctness and smoke-serving baseline for short text prompts. It does not claim optimized throughput, long-context stability, or multimodal correctness.

## Fused SDPA validation

Gemma4 prompt attention was re-enabled for HPU `fsdpa_impl` by removing the Gemma4-specific FSDPA exclusion in `vllm_gaudi/extension/features.py`.

Validation logs:

```text
/tmp/gemma4-e2b-forced-fsdpa-probe-20260528T005509Z.log
/tmp/gemma4-31b-fp8-forced-fsdpa-smoke-20260528T005738Z.log
/tmp/gemma4-e2b-default-fsdpa-afterpatch-20260528T010559Z.log
/tmp/gemma4-e2b-default-fsdpa-postcommit-20260528T011949Z.log
/tmp/gemma4-31b-fp8-default-fsdpa-afterpatch-20260528T010906Z.log
```

Observed fused-attention evidence:

- E2B forced/default runs reported `prompt_attn_impl: fsdpa_impl` and preserved the validated Seoul, Intel Gaudi translation, and `8` outputs.
- 31B FP8 forced/default runs reported `prompt_attn_impl: fsdpa_impl` and preserved the validated Seoul and `12` outputs.
- 31B FP8 default FSDPA memory evidence: model weights `32.4903 GB`, KV cache allocation `49.63 GiB`, total HPU use after cache init `82.02 GiB / 94.62 GiB`, and KV cache size `59,008 tokens`.

Remaining risks: long context, multimodal prompts, throughput benchmarking, FSDPA slicing, and chunked prefix-cache stress paths still need separate validation.

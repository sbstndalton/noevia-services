# Phase 1a — GPU Experiments on DaServer: 890M + Arc A380

Two experiments, ~30 minutes total, both run over SSH on DaServer. Record every number
back into `DaServer.md` (backend evaluation table gets a new subsection) — the point is to
settle the A380 question with data, either way.

**Background (from the doc):** the A380's ANV Vulkan driver showed 0 GPU allocation for
llama.cpp compute (tested); Arc SYCL/Level Zero *did* work (9.37 t/s on 9B). The 890M
(RADV Vulkan, ~16.9 GiB UMA+GTT) is the permanent stack via Lemonade.

---

## Experiment 1a-1: Layer split across 890M (RADV) + A380 (ANV)

Question: can one model's layers be split across both GPUs, pooling 16GB + 6GB?

1. Get a llama.cpp Vulkan binary that sees both devices. The Lemonade llama.cpp build
   (`/mnt/docker/appdata/lemonade/bin/`) enumerates all Vulkan devices; verify:
   ```bash
   docker exec -it lemonade /mnt/docker/appdata/lemonade/bin/llama-b9632 --list-devices
   # expect: RADV RX 890M (device 0) + ANV Arc A380 (device 1)
   ```
2. Run a model the 890M fits comfortably alone (Gemma-4-E2B Q8_0) and record the baseline:
   ```bash
   llama-b9632 -m /models/gemma-4-E2B-it-Q8_0.gguf -ngl 99 -p "Hello" -n 128
   # record: t/s, which GPU via radv/anv log lines
   ```
3. Force the split (llama.cpp maps `-ngl` per device by order; use `--split-mode layer`):
   ```bash
   llama-b9632 -m /models/gemma-4-E2B-it-Q8_0.gguf -ngl 99 --split-mode layer \
     --tensor-split 7,3 -p "Hello" -n 128
   ```
4. **Read the log lines carefully.** If ANV shows `GGML_VULKAN: Device 1 initialization:`
   followed by compute-allocation failures or silent CPU fallback, the split is
   *theoretical only* — layers on the A380 run on CPU and t/s will crater.
5. Repeat with GPT-OSS-20B MXFP4 (the model that would actually benefit — does it fit
   or run faster with the extra 6GB of layer headroom?).

**Adopt if:** split t/s ≥ 890M-only t/s AND stable across 3 runs AND it enables a model
the 890M alone cannot hold. **Otherwise:** close the question permanently with the numbers.

## Experiment 1a-2: A380 as auxiliary-model server (SYCL)

The realistic win: run the diary agent's small aux models (skip-classifier, summarizer —
Gemma-4-E2B class) on the A380 via SYCL, so the 890M is never interrupted mid-conversation.

1. Run a second llama.cpp server built with SYCL on the A380 (level-zero backend):
   ```bash
   # inside a container with Intel compute runtime + oneDNN SYCL llama.cpp build:
   GGML_SYCL_DEVICE=0 llama-server -m /models/gemma-4-E2B-it-Q8_0.gguf -ngl 99 --port 13306
   ```
   (Verify the A380 is the only SYCL device visible: `sycl-ls` should list `level_zero:0`.)
2. Sanity-check throughput: `curl http://10.69.0.130:13306/v1/chat/completions ...`
   Expect E2B-class speed roughly in the 15–40 t/s range from the doc's SYCL numbers.
3. A/B against aux-on-890M: measure the diary agent's aux-call latency
   (`config.yaml llm.aux.base_url` → 13305 vs 13306) during an active 890M chat.

**Adopt if:** aux latency on the A380 ≤ aux latency on the 890M-while-busy. Either result,
log it — this decides whether `docker-compose` grows an aux-model sidecar.

---

## Recording results

Append to the Changelog in DaServer.md, e.g.:

> | 2026-09-XX | GPU experiments: 890M+A380 Vulkan layer-split — <result + numbers>; A380 SYCL aux offload — <adopted/rejected + latency numbers>. | Settles the A380 pooling question with data; AI-stack unchanged unless adopted. |

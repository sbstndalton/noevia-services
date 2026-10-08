# Private Laya decision service

CPU-only typed decisions for Noevia, independent of the answering provider. This is
not an OpenAI chat-completions server. Answering models still use Noevia's existing
provider registry, including OpenRouter and OpenAI-compatible endpoints.

Runtime: Laya 0.3.5, CPU PyTorch 2.14.0, Transformers 5.17.0. The English root model
is pinned to Hugging Face revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
Source: https://huggingface.co/convaiinnovations/laya and
https://github.com/NandhaKishorM/laya . Apache-2.0 upstream license.

`download.py` is an explicit provisioning operation; serving cannot download or
switch weights. Download the five checkpoint/config/tokenizer files into `/model`
using a separate, authorized provisioning container with network access. The serving
container mounts the model read-only, sets the Hugging Face runtime offline, and
uses an internal network shared only with web. No host port or GPU access.
Compose profile `laya`: two CPUs, 6 GiB memory, 128 PIDs, non-root, read-only root.
The optional Docker build arguments `RUNTIME_IMAGE` and `REUSE_CPU_RUNTIME=true`
permit reusing an already-installed compatible CPU runtime; DaServer uses the existing
Docling image as a build base without modifying or restarting Docling.

Set `COWORK_LAYA_MODEL_DIR` to the absolute model directory and
`COWORK_DECISION_URL=http://laya:8040` for web. Leave
`NOEVIA_FEATURE_STEP_SUPERVISION` empty to permit the existing admin switch.
The switch defaults off. Setting that variable to false locks it off.
Provision/start using the existing host preflight with `--profile laya` and service
`laya`; follow the live release runbook rather than replacing its Compose file.

POST `/v1/decisions` accepts `{state, question, options:[{id,label}]}` and returns
`{selected,scores,model,calibrated:false}`. GET `/health` checks worker availability
without inference. The single worker has a 90-second startup deadline and a
request deadline of 1.3 seconds by default. Set `LAYA_DECISION_TIMEOUT_S`
(seconds, 0.5 to 2.0; unset keeps 1.3) to change it. A garbage or out-of-range value is
refused with a startup log line and the default is used. The web decision deadline
(Settings, at most 2000 ms) must stay at or above this value so web receives Laya's
answer; the recommended pair is Laya 1.8 s with web 2000 ms. A request that misses the deadline gets a 503, but the
worker is kept for 15 seconds so the late answer (discarded, never returned) can finish; at
most two such answers may be outstanding, then the worker counts as hung (noevia#1070: killing
it cost a ~17 s reload and left a cold replacement). While idle, the worker reads its weights
every `LAYA_KEEP_WARM_S` seconds (0 to 3600, 0 off, default 30; no inference, no request data)
so the host does not swap them out and make the first decision after a quiet spell miss its
deadline. A crashed, failed or hung worker is terminated,
and one replacement loads in the background. Health reports unready during loading;
the failed decision is never replayed. Replacement startup uses the same 90-second
deadline with at most three attempts and short backoff. If all attempts fail, the
parent exits nonzero so the Compose `on-failure:3` policy can surface a persistent
failure. No request body is logged.
Over-budget inputs are rejected before inference. No tools or approval authority
are exposed to the service.

The Noevia adapter sends a bounded projection: 300 goal characters and the last
150 characters of up to three explicit outputs. It validates the complete probability
map, winning choice and response size, and disables redirects. Failure falls back
to the current chat behavior. This deliberately small projection is not the canonical
record and can omit relevant evidence. Root Laya is English-only; its suitability
for this checkpoint policy is unqualified. Model probabilities are not established
correctness probabilities. No quality or speed claims follow from the smoke test.

Verification: Python validation tests use no model imports; Node tests use mocked
HTTP responses and answering providers. The one authorized real synthetic smoke test
on DaServer returned `continue` with scores 0.465 / 0.2975 / 0.2374. No benchmark,
training, paid endpoint call, or real Diary/personal source test was performed.
As of 2026-09-22, Settings → Server → Experimental provides shared endpoint setup.
Both System-One routing and Step supervision can use Laya. Saving configuration there
overrides the initial COWORK_DECISION_URL default without a restart. The legacy
option-logit endpoint remains a compatibility fallback when no shared endpoint is set.

## Concurrency (`LAYA_MAX_CONCURRENCY`, issue #780)

The HTTP server is threaded (daemon threads); `/health` never waits on inference.
`LAYA_MAX_CONCURRENCY` (integer 1 to 4, default 1, garbage refused with a log line)
sets the number of inference slots. Each slot is its own worker process with its own
model copy and pipe, so a request holds one slot exclusively and nothing mutable is
shared. Requests beyond the bound queue for `LAYA_QUEUE_TIMEOUT_MS` (0 to 10000,
default 2000, the web deadline maximum, so queueing matches the old single-threaded
server), then get `503` with `Retry-After: 1` (web already fails closed).

Thread safety: the model is deliberately not shared across threads. Laya 0.3.5
`Agent.system_one` is `@torch.no_grad()` over an `eval()` model and its router notes
that concurrent predictions may share a checkpoint, but `Agent` mutates `self.device`
and `self.dtype` in its GPU out-of-memory fallback (not reachable on CPU), the
HuggingFace tokenizer is called from the same object, and `torch.set_num_threads(2)`
is process-wide. Per-process slots avoid all of that.

Cost: every extra slot loads another model copy (RSS not yet measured; the container
limit is 6 GiB and 2 CPUs, which also caps real speedup). Raising the bound, and
`NOEVIA_ROUTER_CHUNKS`, needs a measured, owner-approved run.

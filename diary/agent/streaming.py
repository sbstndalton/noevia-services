"""Request-local progress stream around the single journaled exchange."""
import asyncio
import json
import logging
import queue
import threading
from fastapi.responses import StreamingResponse

from .deadline import ExchangeTimeout
from .llm import EMPTY_ANSWER_TEXT, LENGTH_CAP_TEXT, EmptyAnswerError, LengthCapError
from .temporal_native import TemporalUnavailable, UNAVAILABLE_TEXT as TEMPORAL_UNAVAILABLE_TEXT

log = logging.getLogger("diary.streaming")

# #1505: the companion's answer comes before any save, so a deadline passing while it was still
# generating means nothing was written. The code is the one noevia core and web already map for a
# model failure that saved nothing (saved:false keeps them from rewording or restoring it as unknown).
MODEL_TIMEOUT_TEXT = "The local model did not answer in time — nothing was saved. You can resend."


def describe_failure(exc):
    """Exception types and an HTTP status only: provider bodies, request URLs, prompts and diary text
    never reach the log (LLMError and httpx messages can carry them)."""
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen and len(chain) < 4:
        seen.add(id(exc))
        status = getattr(getattr(exc, "response", None), "status_code", None)
        chain.append(type(exc).__name__ + (f" status={status}" if status else ""))
        exc = exc.__cause__ or exc.__context__
    return " <- ".join(chain)


def exchange_stream(work):
    events = queue.Queue(maxsize=256)
    closed = threading.Event()

    def emit(event):
        # #1505: a model's reasoning is never shown to the user, whatever produced it.
        if isinstance(event, dict) and event.get("type") == "reasoning":
            return
        while not closed.is_set():
            try:
                events.put(event, timeout=0.1)
                return
            except queue.Full:
                pass

    def run():
        try:
            result = work(emit)
            emit({"type": "diary", **result})
            emit({"type": "done"})
        except TemporalUnavailable:
            emit({"type": "error", "code": "contextUnavailable", "saved": False,
                  "text": TEMPORAL_UNAVAILABLE_TEXT})
        except ExchangeTimeout as exc:
            log.error("Diary exchange failed: %s", describe_failure(exc))
            if exc.nothing_saved:
                # #127: `reason` tells a deadline apart from other model failures; `code` stays for core and web.
                emit({"type": "error", "code": "modelFailed", "reason": "chat_timeout", "saved": False, "text": MODEL_TIMEOUT_TEXT})
            else:
                emit({"type": "error", "text": "Diary processing failed. Saving may be incomplete; check the saved entry before sending again."})
        except LengthCapError as exc:
            log.error("Diary exchange failed: %s", describe_failure(exc))
            # #135: the answer cap, not a model fault; nothing was saved. `code` stays for core and web.
            emit({"type": "error", "code": "modelFailed", "reason": "length_cap", "saved": False, "text": LENGTH_CAP_TEXT})
        except EmptyAnswerError as exc:
            log.error("Diary exchange failed: %s", describe_failure(exc))
            # #146: a normal stop with no answer text; nothing was saved and no diary event follows.
            emit({"type": "error", "code": "modelFailed", "reason": "empty_answer", "saved": False, "text": EMPTY_ANSWER_TEXT})
        except Exception as exc:
            log.error("Diary exchange failed: %s", describe_failure(exc))
            # Never expose provider HTML, credentials or private prompt bodies.
            emit({"type": "error", "text": "Diary processing failed. Saving may be incomplete; check the saved entry before sending again."})
        finally:
            emit(None)

    async def stream():
        threading.Thread(target=run, daemon=True).start()
        try:
            yield 'data: {"type":"status","text":"Opening diary context…"}\n\n'
            while True:
                try:
                    event = await asyncio.to_thread(events.get, True, 5)
                except queue.Empty:
                    yield ': keep-alive\n\n'
                    continue
                if event is None:
                    break
                yield 'data: ' + json.dumps(event) + '\n\n'
        finally:
            # The journaled operation can finish after a disconnect. Never retry
            # it implicitly; discard further UI events rather than block saving.
            closed.set()

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
    })

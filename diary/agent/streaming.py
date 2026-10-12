"""Request-local progress stream around the single journaled exchange."""
import asyncio
import json
import logging
import queue
import threading
from fastapi.responses import StreamingResponse

log = logging.getLogger("diary.streaming")


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

"""Request-local progress stream around the single journaled exchange."""
import asyncio
import json
import queue
import threading
from fastapi.responses import StreamingResponse


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
        except Exception:
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

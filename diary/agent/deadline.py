"""#1505: a deadline for one Diary exchange, so a stuck model call cannot hold the run open for minutes.

A call runs on a helper thread and the exchange waits for it within the budget. When the budget
runs out the exchange stops waiting and raises ExchangeTimeout. The helper cannot be killed, so it is
abandoned: anything it emits afterwards is dropped, and it ends by its own socket timeouts.

Only the companion's answer and the steps before its save are bounded. Once saving has begun, the
save itself is never interrupted; a failed aux step there falls back the way its existing failure does.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

# Seconds. Total budget for one exchange's companion answer and the decisions made before its save
# (classifier, summary). A slow local model can spend most of this on prompt processing alone.
EXCHANGE_DEADLINE_S = 300.0
# Seconds with no sign of life from the companion model before its answer is given up. Any provider
# stream line counts as life, so a long reasoning phase does not trip it.
FIRST_TOKEN_DEADLINE_S = 150.0
# Seconds for any single auxiliary call (classifier, summary, standing-section maintenance).
AUX_STEP_DEADLINE_S = 60.0


class ExchangeTimeout(RuntimeError):
    """The exchange passed its deadline; the call was abandoned rather than waited for.

    `nothing_saved` is set by the caller that knows no save had begun when it was raised (the
    companion's answer always precedes the save), so the user can be told nothing was written."""

    nothing_saved = False


class _Abandoned(BaseException):
    """Raised inside an abandoned call so it stops emitting and retrying."""


class ExchangeBudget:
    def __init__(self, total_s=None, first_token_s=None, clock=time.monotonic):
        self._clock = clock
        self._end = clock() + (EXCHANGE_DEADLINE_S if total_s is None else total_s)
        self._first_token_s = FIRST_TOKEN_DEADLINE_S if first_token_s is None else first_token_s

    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    def run(self, call, *, emit=None, first_token=False, cap=None):
        """Run call(gate, alive) and return its value, or raise ExchangeTimeout within the budget.

        `gate` is the emit to hand to the call: events reach `emit` only while the call is still
        waited for. `alive()` is called by the call on provider progress. With first_token=True the
        call must show progress within FIRST_TOKEN_DEADLINE_S. `cap` limits this one call further.
        """
        total = self.remaining() if cap is None else min(self.remaining(), cap)
        if total <= 0:
            raise ExchangeTimeout("the exchange deadline passed before the companion was called")
        started, finished, abandoned = threading.Event(), threading.Event(), threading.Event()
        box = {}

        def alive():
            # An abandoned call that is still reading the provider stops at its next sign of life,
            # the same way an abandoned emit does, instead of streaming to the end unobserved.
            if abandoned.is_set():
                raise _Abandoned()
            started.set()

        def gate(event):
            if abandoned.is_set():
                raise _Abandoned()
            started.set()
            if emit is not None:
                emit(event)

        def target():
            try:
                box["value"] = call(gate, alive)
            except _Abandoned:
                pass
            except BaseException as exc:  # noqa: BLE001 - re-raised in the waiting thread
                box["error"] = exc
            finally:
                finished.set()
                started.set()

        threading.Thread(target=target, name="diary-exchange-call", daemon=True).start()
        began = self._clock()
        if first_token and not started.wait(min(self._first_token_s, total)):
            abandoned.set()
            log.warning("Diary companion sent no first token within %.0fs; giving up on it", min(self._first_token_s, total))
            raise ExchangeTimeout("no first token from the companion before its deadline")
        if not finished.wait(max(0.0, total - (self._clock() - began))):
            abandoned.set()
            log.warning("Diary companion call passed its %.0fs deadline; giving up on it", total)
            raise ExchangeTimeout("the companion did not finish before its deadline")
        if "error" in box:
            raise box["error"]
        return box.get("value")

"""Log pipeline — decides whether/how each exchange is logged, then logs it durably.

Decision sequence per exchange:
  1. skip_classifier (aux LLM): LOG or SKIP — meta/administrative exchanges are skipped.
  2. summarizer (aux LLM): assistant reply -> third-person prose for the assistant field.
  3. log_exchange: write-ahead journal + conditional backend append (idempotent replay).
  4. standing-section maintenance (gated, cheap aux call): INDEX.md update only when warranted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from .corpus_store import CorpusStore
from .llm import LLMClient

log = logging.getLogger(__name__)

THINKING_MODEL_HINT = (
    "Output must be the final answer only. Do not show reasoning; do not wrap the answer in "
    "quotes or code fences."
)


@dataclass
class LogOutcome:
    decision: str           # 'logged' | 'skipped' | 'error'
    xid: Optional[str]
    reason: str


class LoggingPipeline:
    def __init__(self, store: CorpusStore, llm_main: LLMClient, llm_aux: LLMClient, templates: dict):
        self.store = store
        self.llm_main = llm_main
        self.llm_aux = llm_aux
        self.templates = templates

    # ---------------- LLM template calls (aux server) ----------------

    def _aux_call(self, template_key: str, system_prompt: str, **fields) -> str:
        prompt = self.templates[template_key].format(**fields)
        return self.llm_aux.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
            stop=None,
        )

    # ---------------- steps ----------------

    def classify(self, user_message: str, assistant_message: str) -> bool:
        """True => log, False => skip."""
        try:
            verdict = self._aux_call(
                "skip_classifier",
                THINKING_MODEL_HINT,
                user_message=user_message,
                assistant_message=assistant_message,
            )
        except Exception as exc:  # noqa: BLE001 — classifier failure must never lose a diary entry
            log.warning("skip classifier failed (%s); defaulting to LOG", exc)
            return True
        return "SKIP" not in verdict.upper()[:12]

    def summarize(self, assistant_message: str) -> str:
        try:
            out = self._aux_call("summarizer", THINKING_MODEL_HINT, assistant_message=assistant_message)
            prose = out.strip()
            if prose.startswith('"') and prose.endswith('"') and len(prose) >= 2:
                prose = prose[1:-1].strip()
            return prose
        except Exception as exc:  # noqa: BLE001 — summary failure must not drop the exchange
            log.error("summarizer failed (%s); logging verbatim assistant reply", exc)
            return assistant_message.strip()

    def maintain_index(self, user_message: str, assistant_summary: str, today: str) -> bool:
        """Gate then edit INDEX.md standing sections. Returns True if INDEX.md changed."""
        if not getattr(self.store, "index_enabled", True):
            return False  # corpus runs without an INDEX.md — skip the gate call entirely
        try:
            import json as _json

            current_sections = self.store.get_standing_sections_text(max_chars=4000)
            gate = self._aux_call(
                "index_maintenance",
                THINKING_MODEL_HINT,
                user_message=user_message,
                assistant_summary=assistant_summary,
                current_sections=current_sections,
            ).strip().upper()
            if not gate.startswith("UPDATE"):
                return False
            raw = self._aux_call(
                "index_edit",
                THINKING_MODEL_HINT,
                user_message=user_message,
                assistant_summary=assistant_summary,
                current_sections=current_sections,
                today=today,
            )
            data = _json.loads(_json_extract(raw))
            changed = self.store.update_standing_sections(
                data.get("open_questions") or [], data.get("timeline") or [], today
            )
            return changed is not None
        except Exception as exc:  # noqa: BLE001
            log.warning("index maintenance failed: %s", exc)
            return False

    # ---------------- entry point ----------------

    def log_exchange(
        self,
        user_message: str,
        assistant_message: str,
        topic: str = "",
        now: Optional[datetime] = None,
        day: Optional[date] = None,
    ) -> LogOutcome:
        now = now or datetime.now()
        day = day or now.date()

        if not self.classify(user_message, assistant_message):
            return LogOutcome(decision="skipped", xid=None, reason="meta/administrative or non-substantive")

        summary = self.summarize(assistant_message)
        try:
            xid = self.store.log_exchange(
                day=day,
                sub_header=topic,
                me_text=user_message,
                claude_text=summary,
                now=now,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("durable logging failed")
            return LogOutcome(decision="error", xid=None, reason=str(exc))

        # Standing sections: opportunistic; failures never affect the logged exchange.
        self.maintain_index(user_message, summary, day.isoformat())
        return LogOutcome(decision="logged", xid=xid, reason="")

    def relog_last(self, user_message: str, assistant_message: str, topic: str = "", now: Optional[datetime] = None) -> LogOutcome:
        """Manual re-log button — bypasses the skip classifier only; idempotent via journal+markers."""
        now = now or datetime.now()
        summary = self.summarize(assistant_message)
        try:
            xid = self.store.log_exchange(
                day=now.date(),
                sub_header=topic,
                me_text=user_message,
                claude_text=summary,
                now=now,
            )
            return LogOutcome(decision="logged", xid=xid, reason="manual re-log")
        except Exception as exc:  # noqa: BLE001
            return LogOutcome(decision="error", xid=None, reason=str(exc))


def _json_extract(text: str) -> str:
    """Extract the first JSON object from a model reply (defensive against prose/fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    return text[start : end + 1]

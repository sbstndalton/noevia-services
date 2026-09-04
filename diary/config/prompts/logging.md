# Diary Companion — logging behavior rules (templates for the log-pipeline LLM calls)
# All templates take the system prompt from config/prompts/system.md as the system message.

# Does this exchange contain substantive content (thoughts/feelings/events) or is it meta/administrative?
skip_classifier: |
  Two speaker turns follow. Decide whether the exchange contains substantive diary content —
  thoughts, feelings, or events from the user's life — or is meta/administrative (about the
  diary's own structure, formatting, mechanics, or this system's behavior, or trivial chit-chat
  with nothing substantive).

  Answer with exactly one word: LOG or SKIP.
  LOG if in doubt.

  User: {user_message}
  Assistant: {assistant_message}
  Verdict (LOG or SKIP):

# Condense the assistant's reply into third-person natural prose for the diary. Not bullets.
summarizer: |
  Convert the assistant's diary-companion reply below into third-person natural prose suitable
  for a permanent diary, as if a careful biographer recorded the exchange. Requirements:
  - Prose paragraphs only. Never bullet points, never headers, never lists.
  - Keep every substantive fact, feeling, pattern, and commitment; drop conversational filler.
  - Preserve the assistant's honest assessments and any gentle challenge it offered — do not
    soften it into empty validation.
  - Attribute the assistant's points to "the assistant" or "the diary companion"; attribute the
    user's content to "the user" only if you must reference it; the user's own words are logged
    verbatim elsewhere, so do not restate them at length.
  - If the assistant checked in on safety, record that it did so and the concern, plainly.
  - Output ONLY the prose. No preamble, no quotes around it.

  Assistant reply to convert:
  {assistant_message}

# Does this exchange warrant updating a standing section (open questions / timeline of key events)?
index_maintenance: |
  You maintain the standing sections of a diary index: "Open Questions" (unresolved questions
  the user wants revisited) and "Timeline of Key Events" (significant life events with dates).

  Latest diary exchange:
  User: {user_message}
  Assistant: {assistant_summary}

  Current standing sections:
  {current_sections}

  Does this exchange warrant adding, resolving, or updating any item in either section?
  Most exchanges do not. Answer with exactly one word: UPDATE or NO.
  UPDATE only if a genuinely new open question or key event appears, or an existing one resolves.

# Suggest concrete INDEX.md edits when the maintenance gate says UPDATE.
index_edit: |
  You maintain the standing sections of a diary index. Given the latest diary exchange and the
  current INDEX.md standing sections, output the minimal edit instructions.

  Latest exchange:
  User: {user_message}
  Assistant: {assistant_summary}

  Current standing sections:
  {current_sections}

  Today's date: {today}

  Output strict JSON, nothing else:
  {{"open_questions": [{{"action": "add"|"resolve"|"edit", "text": "..."}}],
    "timeline": [{{"action": "add"|"edit", "date": "YYYY-MM-DD", "text": "..."}}]}}
  Omit a key if no changes to it. Empty lists if no changes. Keep texts to one sentence.

#!/usr/bin/env python3
"""Deterministic synthetic Diary fixture for the Diary gate.

Everything written here is invented: a fictional cartographer, fictional people and places.
No credentials, no real diary text. The same arguments give byte-identical output.

    python3 -I diary/gate/gen_fixture.py --seed 20261020 --anchor 2030-01-06 --days 60 --tz UTC --out DIR

Output (DIR):
    fixture/Entries/YYYY/Month/Month D, YYYY.md   the diary's daily layout (corpus.entry_layout: daily)
    truth/entries.jsonl                          one row per entry: id, xid, day, path, header, kind
    truth/queries.jsonl                          ground-truth queries
    truth/mutations.jsonl                        scripted paraphrases, corrections and deletions
    manifest.json                                sha256 per file, tree hash, seed, anchor, query clock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

SCHEMA = "diarygate-fixture/1"
NS = uuid.UUID("6f1d3c52-8a0e-4b7a-9c61-2f4e0d9b7a11")
OWNER = "Marit"
SLOTS = ((7, 9), (12, 14), (18, 21))  # hour ranges for the three entries of each day
CORRECTION_HOUR = 22

FIRST = ["Alder", "Bryony", "Caspian", "Delphine", "Evander", "Fenella", "Gideon", "Hesper", "Ivo", "Jessamy",
         "Kestrel", "Lucan", "Mirela", "Nyle", "Orla", "Peregrine", "Quill", "Rosalind", "Soren", "Tamsin"]
# Near-duplicate surname pairs: the needle holder takes the first, its distractor the second.
SURNAMES = [("Brannock", "Brennock"), ("Castellow", "Castelow"), ("Dunmore", "Dunmoore"), ("Elphick", "Elphicke"),
            ("Fairweather", "Fairwether"), ("Garroway", "Garraway"), ("Halloran", "Holloran"), ("Ingleby", "Ingelby"),
            ("Juxon", "Juxton"), ("Kittredge", "Kitteridge"), ("Lockyer", "Lockyear"), ("Mabery", "Maybery"),
            ("Nethercott", "Nethercot"), ("Ormsby", "Ormesby"), ("Pennick", "Penick"), ("Quarrie", "Quarry"),
            ("Rendell", "Rendall"), ("Stanforth", "Stamforth"), ("Trevelyan", "Trevellyan"), ("Uttley", "Utley"),
            ("Vowles", "Voles"), ("Wenlock", "Wendlock"), ("Yardley", "Yeardley"), ("Ashdown", "Ashdowne"),
            ("Blakemore", "Blackmore"), ("Corrigan", "Carrigan"), ("Darnell", "Darnall"), ("Esterby", "Easterby"),
            ("Fennimore", "Fenimore"), ("Gilchrist", "Gillchrist"), ("Hartigan", "Hartagan"), ("Ivensey", "Ivensay"),
            ("Jephcott", "Jepcott"), ("Kilbride", "Killbride"), ("Larkham", "Larkam"), ("Merriman", "Merryman"),
            ("Northam", "Northum"), ("Oakeshott", "Okeshott"), ("Prideaux", "Prideau"), ("Rookwood", "Rockwood")]
# (attribute, question form, value pool). Each attribute is used by two needles.
ATTRS = [
    ("favourite tea", "What is {h}'s favourite tea?", ["smoked lapsang", "nettle and mint", "white peony", "roasted barley", "lemon verbena", "chicory root"]),
    ("bicycle colour", "What colour is {h}'s bicycle?", ["mustard yellow", "racing green", "dusty pink", "petrol blue", "burnt orange", "slate grey"]),
    ("allotment plot", "Which allotment plot does {h} rent?", ["plot 14", "plot 27", "plot 33", "plot 41", "plot 58", "plot 62"]),
    ("dog's name", "What is the name of {h}'s dog?", ["Pumpernickel", "Biscotti", "Marzipan", "Gingersnap", "Crumpet", "Strudel"]),
    ("choir part", "Which part does {h} sing in the choir?", ["second alto", "first tenor", "baritone", "mezzo", "bass two", "descant"]),
    ("boat name", "What is {h}'s boat called?", ["the Curlew", "the Whimbrel", "the Dunlin", "the Godwit", "the Knot", "the Sanderling"]),
    ("chess opening", "Which chess opening does {h} always play?", ["the Catalan", "the Scandinavian", "the Pirc", "the Alekhine", "the Benko", "the Trompowsky"]),
    ("weaving pattern", "Which weaving pattern is {h} working on?", ["herringbone twill", "goose eye", "monk's belt", "rosepath", "huckaback", "summer and winter"]),
    ("ferry crossing", "Which ferry crossing does {h} take on Thursdays?", ["the Selby crossing", "the Hamble crossing", "the Orrin crossing", "the Tarn crossing", "the Vell crossing", "the Moss crossing"]),
    ("apple variety", "Which apple variety does {h} grow?", ["Ribston Pippin", "Egremont Russet", "Ashmead's Kernel", "Blenheim Orange", "Pitmaston Pineapple", "Claygate Pearmain"]),
    ("lighthouse shift", "Which lighthouse shift does {h} cover?", ["the dawn watch", "the dusk watch", "the middle watch", "the forenoon watch", "the dog watch", "the morning watch"]),
    ("violin maker", "Who made {h}'s violin?", ["a workshop in Cremsby", "a workshop in Lutherford", "a workshop in Bowness Vale", "a workshop in Fiddleham", "a workshop in Rosinby", "a workshop in Scrollbridge"]),
    ("bread starter", "What does {h} call their bread starter?", ["Old Gregor", "Auntie Rye", "The Brute", "Sir Bubbles", "Mother Hen", "Captain Crust"]),
    ("telescope", "What kind of telescope does {h} use?", ["a 6-inch refractor", "an 8-inch dobsonian", "a 5-inch maksutov", "a 10-inch newtonian", "a 4-inch apochromat", "a 12-inch truss dobsonian"]),
    ("map projection", "Which map projection does {h} prefer?", ["the Winkel tripel", "the Robinson", "the Mollweide", "the Goode homolosine", "the Eckert IV", "the Dymaxion"]),
    ("rowing seat", "Which seat does {h} row in the club eight?", ["stroke seat", "bow seat", "seat three", "seat five", "seat six", "seat two"]),
    ("beekeeping hive", "What type of hive does {h} keep?", ["a Warre hive", "a top-bar hive", "a National hive", "a Langstroth hive", "a skep", "a Layens hive"]),
    ("kite", "What kind of kite does {h} fly?", ["a box kite", "a rokkaku", "a sled kite", "a delta kite", "a parafoil", "a diamond kite"]),
    ("birthday month", "In which month is {h}'s birthday?", ["in March", "in August", "in November", "in May", "in October", "in February"]),
    ("sea swim spot", "Where does {h} swim in the sea?", ["Gullet Cove", "Pebbleworth Bay", "Saltmarsh Point", "Whelk Strand", "Cormorant Steps", "Driftwood Reach"]),
]
PLACES = ["the harbour market", "the map library", "the reading room", "the tide gauge", "the co-op bakery",
          "the county archive", "the boatyard", "the station cafe", "the print shop", "the survey office"]
FILLER_ACTIVITIES = ["redrawing the contour lines on the estuary sheet", "inking the legend for the coastal chart",
                     "scanning old field notebooks", "sorting the plan chest", "walking the cliff path with a measuring wheel",
                     "checking benchmark heights near the quarry", "cleaning the drafting pens", "re-lettering the parish boundaries",
                     "photographing the tithe maps", "tidying the studio shelves", "planning the next field day",
                     "colour-matching the moorland tints", "writing up the hedge survey", "testing a new tracing film",
                     "re-checking grid references on the river sheet", "labelling the drawer of aerial prints"]
FILLER_MOODS = ["Felt calm and focused.", "Tired by the afternoon.", "A slow but satisfying day.", "Restless, could not settle.",
                "Pleased with the progress.", "Frustrated by the light.", "Quietly content.", "Distracted but got there."]
WEATHER = ["Drizzle all morning.", "Bright and cold.", "Fog rolled in off the water.", "Strong westerly wind.",
           "Mild and overcast.", "Clear skies by evening.", "Sleet showers.", "Still and grey."]
# Multi-entry series: a survey project whose three stops are recorded in three different entries.
SERIES = [("Hollin Moor", ["Thwaite End", "Scarrow", "Gillamoor Cross"]), ("Beckside", ["Nab Crag", "Lowfold", "Ravenscar Hill"]),
          ("Saltings", ["Marram Point", "Cockle Sands", "Reedby"]), ("Upper Vale", ["Hagg Wood", "Starbotton Rise", "Kettlewick"]),
          ("Westmere", ["Fenwick Lock", "Otterburn Mill", "Sedgeholme"]), ("Coppice", ["Birkdale Hollow", "Ashby Fold", "Thornrigg"]),
          ("Estuary", ["Mudlark Quay", "Samphire Flats", "Heronsgate"]), ("Highfell", ["Cairn Brow", "Windy Gap", "Tarnside"])]
# Ambiguous: two different people who share a first name, each giving Marit something different.
AMBIGUOUS = [("Tobiah", ("the baker", "a jar of quince jelly"), ("the ferry pilot", "a brass compass")),
             ("Wren", ("the archivist", "a box of glass slides"), ("the potter", "a celadon bowl")),
             ("Hollis", ("the beekeeper", "a comb of heather honey"), ("the bus driver", "a folding stool")),
             ("Juniper", ("the surveyor", "a spare plumb bob"), ("the florist", "a bunch of sweet peas"))]
# Never recorded anywhere in the fixture (a test checks each topic word is absent from every entry).
UNANSWERABLE = [
    ("What is the name of the school I went to as a child?", "school"),
    ("What was the final score of the regional curling match I watched?", "curling"),
    ("Which dentist did I see in February?", "dentist"),
    ("What did I pay for the new drafting table?", "drafting table"),
    ("What is the phone model of my cousin Ottoline?", "Ottoline"),
    ("Which film did I see at the cinema on my birthday?", "cinema"),
]


def day_path(d: date) -> str:
    return f"Entries/{d.year}/{d.strftime('%B')}/{d.strftime('%B')} {d.day}, {d.year}.md"


def day_header(d: date) -> str:
    return f"## {d.strftime('%A')}, {d.strftime('%B')} {d.day}, {d.year}"


def render_block(header: str, me: str, assistant: str, xid: str) -> str:
    # agent/corpus.py render_subsection_block(header, render_exchange(...)) without its final newline.
    return f"### {header}\n\n**Me:** {me}\n\n**Assistant:** {assistant}\n\n<!-- xid:{xid} -->"


def render_day(d: date, blocks: list) -> str:
    # agent/corpus.py append_to_month_text: day header, blank line, blocks separated by one blank line.
    return day_header(d) + "\n\n" + "\n\n".join(blocks) + "\n"


def tree_hash(files: dict) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\0{files[rel]}\n".encode())
    return h.hexdigest()


def generator_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build(seed: int, anchor: date, days: int):
    if days < 30:
        raise SystemExit("--days must be at least 30 (early/middle/late thirds need room)")
    rng = random.Random(seed)

    def xid_of(key):
        return str(uuid.uuid5(NS, f"{seed}:{key}"))

    all_days = [anchor + timedelta(days=i) for i in range(days)]
    third = days // 3
    buckets = {"early": all_days[:third], "middle": all_days[third:2 * third], "late": all_days[2 * third:]}
    free = {(d, s) for d in all_days for s in range(len(SLOTS))}

    def take(candidate_days, avoid_days=()):
        allowed, avoid = set(candidate_days), set(avoid_days)
        options = sorted(x for x in free if x[0] in allowed and x[0] not in avoid)
        if not options:
            options = sorted(x for x in free if x[0] not in avoid)
        slot = rng.choice(options)
        free.discard(slot)
        return slot

    placed = []
    names = rng.sample(FIRST, len(FIRST))
    pairs = rng.sample(SURNAMES, len(SURNAMES))
    positions = ["early", "middle", "late"]
    needles = []
    for i in range(40):
        attr, question, pool = ATTRS[i % len(ATTRS)]
        # Needles i and i + 20 share an attribute; give them different first names too.
        first = names[(i + (i // len(ATTRS)) * 7) % len(names)]
        sur, dsur = pairs[i]
        holder, dholder = f"{first} {sur}", f"{first} {dsur}"
        value, dvalue = rng.sample(pool, 2)
        pos = positions[i % 3]
        place, dplace = rng.sample(PLACES, 2)
        n_slot = take(buckets[pos])
        d_slot = take(all_days, avoid_days=[n_slot[0]])
        needles.append({"i": i, "attr": attr, "question": question.format(h=holder), "holder": holder, "dholder": dholder,
                        "value": value, "dvalue": dvalue, "pos": pos, "pool": pool})
        placed.append({"slot": n_slot, "kind": "needle", "key": f"needle-{i}", "topic": f"Bumped into {holder}",
                       "me": f"Ran into {holder} at {place} today. We got talking and they told me their {attr} is {value}.",
                       "assistant": f"{OWNER} met {holder} at {place}; {holder} said their {attr} is {value}."})
        placed.append({"slot": d_slot, "kind": "distractor", "key": f"distractor-{i}", "topic": f"Chat with {dholder}",
                       "me": f"Had coffee with {dholder} near {dplace}. Funny coincidence: their {attr} is {dvalue}.",
                       "assistant": f"{OWNER} had coffee with {dholder}, whose {attr} is {dvalue}."})
    # Restatements: four needles are said again later in other words (alternative relevant sets).
    for k in (1, 8, 22, 31):
        n = needles[k]
        nday = next(p["slot"][0] for p in placed if p["key"] == f"needle-{k}")
        slot = take([d for d in all_days if d > nday] or all_days, avoid_days=[nday])
        placed.append({"slot": slot, "kind": "restatement", "key": f"restatement-{k}", "topic": "Follow-up thought",
                       "me": f"Thinking back to {n['holder']}: I remembered it right, the {n['attr']} they mentioned really is {n['value']}.",
                       "assistant": f"{OWNER} confirmed that {n['holder']}'s {n['attr']} is {n['value']}."})
        n["restatement"] = f"restatement-{k}"
    series = []
    for si, (project, stops) in enumerate(SERIES):
        keys = []
        for k, (stop, d) in enumerate(zip(stops, sorted(rng.sample(all_days, 3)))):
            slot = take([d])
            keys.append(f"series-{si}-{k}")
            placed.append({"slot": slot, "kind": "series", "key": keys[-1], "topic": f"{project} survey, stop {k + 1}",
                           "me": f"Day {k + 1} of the {project} survey: measured the boundary stones at {stop}.",
                           "assistant": f"{OWNER} surveyed {stop} as stop {k + 1} of the {project} survey."})
        series.append({"project": project, "stops": stops, "keys": keys})
    ambiguous = []
    for ai, (first, (role_a, gift_a), (role_b, gift_b)) in enumerate(AMBIGUOUS):
        keys = []
        for k, (role, gift) in enumerate(((role_a, gift_a), (role_b, gift_b))):
            slot = take(all_days)
            keys.append(f"ambiguous-{ai}-{k}")
            placed.append({"slot": slot, "kind": "ambiguous", "key": keys[-1], "topic": f"A gift from {first}",
                           "me": f"{first} {role} gave me {gift} today, completely out of the blue.",
                           "assistant": f"{first} {role} gave {OWNER} {gift}."})
        ambiguous.append({"first": first, "gifts": [gift_a, gift_b], "keys": keys})
    for fi, slot in enumerate(sorted(free)):
        act = rng.choice(FILLER_ACTIVITIES)
        placed.append({"slot": slot, "kind": "filler", "key": f"filler-{fi}", "topic": "Studio day",
                       "me": f"Spent the session {act}. {rng.choice(WEATHER)} {rng.choice(FILLER_MOODS)}",
                       "assistant": f"{OWNER} spent the session {act}."})
    free.clear()
    placed.sort(key=lambda p: p["slot"])
    by_key, entries = {}, []
    for n, p in enumerate(placed, 1):
        d, s = p["slot"]
        lo, hi = SLOTS[s]
        hhmm = f"{rng.randint(lo, hi):02d}:{rng.randrange(0, 60):02d}"
        row = {"id": f"E{n:03d}", "xid": xid_of(p["key"]), "day": d.isoformat(), "path": day_path(d),
               "header": f"{hhmm} — {p['topic']}", "kind": p["kind"], "key": p["key"], "me": p["me"], "assistant": p["assistant"]}
        entries.append(row)
        by_key[p["key"]] = row
    # Entries of one day in time order (slot hour ranges do not overlap, so the slot order is the time order).
    files = {}
    for d in all_days:
        blocks = [render_block(e["header"], e["me"], e["assistant"], e["xid"]) for e in entries if e["day"] == d.isoformat()]
        files[day_path(d)] = render_day(d, blocks)

    # ---- scripted mutations (applied through the API after the load) ----
    mutations, mutated, corrections = [], {}, {}
    para_idx, del_idx = (3, 13, 27, 36), (10, 25, 38)
    # A correction is written on a later fixture day, so its needle must be at least two days before the end.
    last_ok = all_days[-3]
    corr_idx = tuple([k for k in (5, 16, 20, 34, 2, 11, 23, 32, 26, 35)
                      if date.fromisoformat(by_key[f"needle-{k}"]["day"]) <= last_ok][:4])
    for k in para_idx:
        n, e = needles[k], by_key[f"needle-{k}"]
        mutations.append({"id": f"M-para-{k}", "op": "edit", "kind": "paraphrase", "entry_id": e["id"], "xid": e["xid"],
                          "month": e["day"][:7],
                          "me": f"{n['holder']} came up in conversation; the {n['attr']} they prefer above all is {n['value']}.",
                          "assistant": f"{OWNER} learned that {n['holder']} prefers {n['value']} as their {n['attr']}."})
        mutated[k] = "paraphrase"
    for c, k in enumerate(corr_idx, 1):
        n, e = needles[k], by_key[f"needle-{k}"]
        nday = date.fromisoformat(e["day"])
        new_value = rng.choice([v for v in n["pool"] if v not in (n["value"], n["dvalue"])])
        cday = max(nday + timedelta(days=1), min(all_days[-2], nday + timedelta(days=rng.randint(3, 9))))
        assert anchor < cday <= all_days[-1]
        header = f"{CORRECTION_HOUR}:{rng.randrange(0, 60):02d} — Correction"
        me = (f"Correction to my note from {nday.strftime('%B')} {nday.day}: {n['holder']}'s {n['attr']} is actually "
              f"{new_value}, not {n['value']}. I had misheard.")
        assistant = f"{OWNER} corrected the earlier note: {n['holder']}'s {n['attr']} is {new_value}."
        xid = xid_of(f"correction-{k}")
        row = {"id": f"C{c:02d}", "xid": xid, "day": cday.isoformat(), "path": day_path(cday), "header": header, "kind": "correction"}
        mutations.append({"id": f"M-corr-{k}", "op": "add_entry", "kind": "correction", "entry_id": row["id"], "xid": xid,
                          "path": row["path"], "block": render_block(header, me, assistant, xid)})
        corrections[k] = (row, new_value)
        mutated[k] = "correction"
    for k in del_idx:
        e = by_key[f"needle-{k}"]
        mutations.append({"id": f"M-del-{k}", "op": "edit", "kind": "deletion", "entry_id": e["id"], "xid": e["xid"],
                          "month": e["day"][:7], "me": "I asked for the detail in this note to be removed.",
                          "assistant": f"{OWNER} withdrew this note."})
        mutated[k] = "deletion"

    # ---- queries ----
    # The query clock is an EMPTY trailing day (day days+1, 12:00Z): no fixture entry is "today", so no
    # needle is pulled into context just for being on the query day; logged exchanges land there too.
    clock_day = all_days[-1] + timedelta(days=1)
    clock = f"{clock_day.isoformat()}T12:00:00Z"

    def cite(*rows):
        return sorted({r["day"] for r in rows})

    def base(qid, category, position, text, rel, expected, forbidden, rows, **extra):
        q = {"id": qid, "category": category, "position": position, "text": text, "valid_as_of": clock_day.isoformat(),
             "clock": clock, "relevant_entry_ids": rel, "expected_facts": expected, "forbidden_facts": forbidden,
             "expect_abstain": False, "ambiguous": False, "mutations": [], "citation_days": cite(*rows)}
        q.update(extra)
        return q

    queries = []
    for n in needles:
        k, e = n["i"], by_key[f"needle-{n['i']}"]
        q = base(f"Q-needle-{k:02d}", "needle", n["pos"], n["question"], [[e["id"]]], [n["value"]], [n["dvalue"]], [e])
        if n.get("restatement"):
            r = by_key[n["restatement"]]
            q.update(relevant_entry_ids=[[e["id"]], [r["id"]]], citation_days=cite(e, r))
        kind = mutated.get(k)
        if kind == "paraphrase":
            q.update(category="paraphrase", mutations=[f"M-para-{k}"])
        elif kind == "correction":
            row, new_value = corrections[k]
            q.update(category="correction", mutations=[f"M-corr-{k}"], relevant_entry_ids=[[row["id"]]],
                     expected_facts=[new_value], forbidden_facts=[n["value"], n["dvalue"]], citation_days=cite(row))
        elif kind == "deletion":
            q.update(category="deletion", mutations=[f"M-del-{k}"], relevant_entry_ids=[], expected_facts=[],
                     forbidden_facts=[n["value"], n["dvalue"]], expect_abstain=True, citation_days=[])
        if k in (0, 7, 14, 19, 29, 33):
            q["followup"] = {"text": f"And what about {n['dholder']}?", "expected_facts": [n["dvalue"]],
                             "forbidden_facts": [n["value"]], "relevant_entry_ids": [[by_key[f'distractor-{k}']['id']]]}
        queries.append(q)
    for k in corr_idx[:2]:
        n, e = needles[k], by_key[f"needle-{k}"]
        row, new_value = corrections[k]
        as_of = date.fromisoformat(row["day"]) - timedelta(days=1)
        queries.append(base(f"Q-asof-{k:02d}", "temporal", n["pos"],
                            f"As of {as_of.strftime('%B')} {as_of.day}, {as_of.year}, what had I written down as {n['holder']}'s {n['attr']}?",
                            [[e["id"]]], [n["value"]], [new_value, n["dvalue"]], [e], valid_as_of=as_of.isoformat(),
                            clock=f"{as_of.isoformat()}T12:00:00Z", mutations=[f"M-corr-{k}"]))
    for si, s in enumerate(series):
        rows = [by_key[x] for x in s["keys"]]
        queries.append(base(f"Q-multi-{si}", "multi-entry", "spread",
                            f"Which three places did I survey during the {s['project']} survey?",
                            [[r["id"] for r in rows]], list(s["stops"]), [], rows))
    for ui, (text, _) in enumerate(UNANSWERABLE):
        queries.append(base(f"Q-unans-{ui}", "unanswerable", "none", text, [], [], [], [], expect_abstain=True))
    for ai, a in enumerate(ambiguous):
        rows = [by_key[x] for x in a["keys"]]
        queries.append(base(f"Q-ambig-{ai}", "ambiguous", "spread", f"What did {a['first']} give me?",
                            [[r["id"] for r in rows]], list(a["gifts"]), [], rows, ambiguous=True))
    entry_rows = [{k: e[k] for k in ("id", "xid", "day", "path", "header", "kind")} for e in entries]
    entry_rows += [row for row, _ in corrections.values()]
    counts = {"entries": len(entries), "days": days, "needles": len(needles), "distractors": len(needles),
              "queries": len(queries), "multi_entry": len(series), "unanswerable": len(UNANSWERABLE),
              "ambiguous": len(ambiguous), "followups": sum(1 for q in queries if "followup" in q),
              "mutations": {"paraphrase": len(para_idx), "correction": len(corr_idx), "deletion": len(del_idx)},
              "positions": {p: sum(1 for n in needles if n["pos"] == p) for p in positions}}
    counts["query_day"] = {"day": clock_day.isoformat(), "entries": 0}
    return files, entry_rows, queries, mutations, {"query_clock": clock, "counts": counts}


def jsonl(rows) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def write(out: Path, seed: int, anchor: date, days: int, tz: str, commit: str, force: bool = False) -> dict:
    files, entries, queries, mutations, meta = build(seed, anchor, days)
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(f"{out} is not empty; pass --force to replace the fixture there")
        for name in ("fixture", "truth"):
            shutil.rmtree(out / name, ignore_errors=True)
        (out / "manifest.json").unlink(missing_ok=True)
    payload = {f"fixture/{p}": t for p, t in files.items()}
    payload["truth/entries.jsonl"] = jsonl(entries)
    payload["truth/queries.jsonl"] = jsonl(queries)
    payload["truth/mutations.jsonl"] = jsonl(mutations)
    hashes = {}
    for rel in sorted(payload):
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = payload[rel].encode("utf-8")
        target.write_bytes(data)
        hashes[rel] = hashlib.sha256(data).hexdigest()
    manifest = {"schema": SCHEMA, "seed": seed, "anchor": anchor.isoformat(), "days": days, "tz": tz,
                "query_clock": meta["query_clock"], "generator": "diary/gate/gen_fixture.py", "generator_commit": commit,
                "counts": meta["counts"], "files": hashes, "tree_sha256": tree_hash(hashes)}
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--anchor", default="2030-01-06", help="first fixture day (ISO date)")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--tz", default="UTC", help="only UTC: entry times and the query clock are UTC (the gate runs TZ=UTC)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generator-commit", default=None, help="override the recorded commit (tests)")
    ap.add_argument("--force", action="store_true", help="replace fixture/, truth/ and manifest.json in --out")
    args = ap.parse_args(argv)
    if args.tz != "UTC":
        raise SystemExit("--tz: only UTC is supported")
    manifest = write(Path(args.out), args.seed, date.fromisoformat(args.anchor), args.days, args.tz,
                     args.generator_commit or generator_commit(), args.force)
    print(json.dumps({"tree_sha256": manifest["tree_sha256"], "files": len(manifest["files"]), "counts": manifest["counts"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

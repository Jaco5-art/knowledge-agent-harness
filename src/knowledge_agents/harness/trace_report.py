
"""Summarize exported Harness JSONL traces without executing agents."""
import argparse
import json
import math
from pathlib import Path

END_EVENTS = {
    "tool_succeeded", "tool_failed",
    "tool_interrupted", "context_budget_exceeded",
}

def metric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None

def summarize(events, session):
    calls = {}
    seen = set()
    unmatched_session = 0
    selected = 0
    review_events = 0

    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Each JSONL record must be an object")
        if not event.get("session_id"):
            unmatched_session += 1
            continue
        if event["session_id"] != session:
            continue

        # Exact duplicate exported events are counted once.
        fingerprint = json.dumps(event, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected += 1
        name = event.get("event")
        if name == "review_required":
            review_events += 1
        if name != "tool_started" and name not in END_EVENTS:
            continue
        cid = event.get("call_id")
        if not cid:
            raise ValueError("Selected tool event has no call_id")
        record = calls.setdefault(cid, {"start": None, "end": None})
        slot = "start" if name == "tool_started" else "end"
        if record[slot] is not None:
            raise ValueError(f"Conflicting {slot} events for call_id {cid}")
        record[slot] = event

    if not selected:
        raise ValueError(
            "No events match this session_id. "
            "Expected flat JSONL records with session_id and event fields."
        )

    rows = []
    for cid, pair in calls.items():
        start, end = pair["start"] or {}, pair["end"] or {}
        rows.append({
            "call_id": cid,
            "agent": start.get("agent"),
            "tool": start.get("tool", end.get("tool")),
            "candidate_id": start.get("candidate_id"),
            "attempt": start.get("attempt", end.get("attempt")),
            "status": end.get("event", "no_terminal_event"),
            "start_present": bool(start),
            "input_chars": metric(start.get("input_chars")),
            "output_chars": metric(end.get("output_chars")),
            "latency_ms": metric(end.get("latency_ms")),
            "input_tokens": metric(end.get("input_tokens")),
            "output_tokens": metric(end.get("output_tokens")),
            "error_type": end.get("error_type"),
        })

    def measured(field):
        values = [r[field] for r in rows if r[field] is not None]
        return {
            "recorded_sum": sum(values) if values else None,
            "recorded_calls": len(values),
            "total_calls": len(rows),
        }

    retry_calls = [
        r for r in rows
        if isinstance(r["attempt"], int)
        and not isinstance(r["attempt"], bool)
        and r["attempt"] > 1
    ]
    return {
        "session_id": session,
        "coverage": "supplied_exported_events_only",
        "selected_unique_events": selected,
        "events_without_session_id": unmatched_session,
        "observed_calls": len(rows),
        "successful_calls": sum(r["status"] == "tool_succeeded" for r in rows),
        "failed_calls": sum(r["status"] == "tool_failed" for r in rows),
        "interrupted_calls": sum(r["status"] == "tool_interrupted" for r in rows),
        "budget_blocked_calls": sum(
            r["status"] == "context_budget_exceeded" for r in rows
        ),
        "calls_without_terminal_event": sum(
            r["status"] == "no_terminal_event" for r in rows
        ),
        "terminal_events_without_start": sum(
            not r["start_present"] for r in rows
        ),
        "observed_retry_calls": len(retry_calls),
        "review_required_events": review_events,
        "latency_ms": measured("latency_ms"),
        "input_tokens": measured("input_tokens"),
        "output_tokens": measured("output_tokens"),
        "cost": None,
        "notes": [
            "Missing measurements are not zero.",
            "Latency sum is not end-to-end wall time.",
            "Calls are Harness tool attempts, not necessarily provider requests.",
            "Cache hits are not counted because no cache-hit event is emitted.",
            "Agent labels are available only when start events are present.",
            "Supplied exports may cover only part of a session.",
        ],
        "calls": rows,
    }

def self_test():
    start = {
        "session_id": "s", "event": "tool_started",
        "call_id": "a", "tool": "extract",
        "agent": "entity_agent", "attempt": 1,
    }
    end = {
        "session_id": "s", "event": "tool_succeeded",
        "call_id": "a", "latency_ms": 2,
        "input_tokens": 10, "output_tokens": 3,
    }
    result = summarize([start, end, start.copy(), end.copy()], "s")
    assert result["observed_calls"] == 1
    assert result["input_tokens"]["recorded_sum"] == 10
    assert result["calls"][0]["agent"] == "entity_agent"

    partial = summarize([start], "s")
    assert partial["calls_without_terminal_event"] == 1
    assert partial["input_tokens"]["recorded_sum"] is None

    retry = dict(end, call_id="b", event="tool_failed", attempt=2)
    result = summarize([start, end, retry], "s")
    assert result["observed_retry_calls"] == 1
    assert result["terminal_events_without_start"] == 1

    other = dict(end, session_id="other", input_tokens=999)
    assert summarize([start, end, other], "s")["observed_calls"] == 1

    try:
        summarize([start, end, dict(end, output_tokens=99)], "s")
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting events were accepted")
    print("SELF TEST OK: trace aggregation checks passed; 0 API calls.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append")
    parser.add_argument("--session")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.trace or not args.session:
        parser.error("Provide --trace FILE --session SESSION_ID")
    events = []
    for path in args.trace:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON: {path.name}, line {line_number}"
                    ) from exc
    result = summarize(events, args.session)
    result["source_files"] = [str(p) for p in args.trace]
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
        print(f"REPORT OK: {args.output.resolve()}")
    else:
        print(text)

if __name__ == "__main__":
    main()

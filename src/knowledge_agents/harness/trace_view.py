"""Local escaped HTML trace view. No external scripts or remote telemetry."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path


def render_trace(source: Path, destination: Path):
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    totals = Counter(e.get("event") for e in events)
    finished = [e for e in events if e.get("event") in {"tool_succeeded", "tool_failed", "tool_interrupted"}]
    rows = []
    for e in events:
        cells = [e.get("state_version"), e.get("stage"), e.get("event"),
                 e.get("candidate_id", ""), e.get("attempt", ""),
                 round(e["latency_ms"], 2) if "latency_ms" in e else "",
                 e.get("input_tokens", ""), e.get("output_tokens", ""),
                 e.get("reason", e.get("error_type", e.get("verdict", "")))]
        rows.append("<tr>" + "".join("<td>" + escape("unknown" if c is None else str(c)) + "</td>" for c in cells) + "</tr>")
    unknown = sum(e.get("input_tokens") is None or e.get("output_tokens") is None for e in finished)
    known_input = sum(e.get("input_tokens") or 0 for e in finished)
    known_output = sum(e.get("output_tokens") or 0 for e in finished)
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Harness Trace</title>
<style>body{font:15px system-ui;margin:32px;color:#182a3a;background:#f6f8fa}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d4dee8;padding:9px;text-align:left;overflow-wrap:anywhere}th{background:#e6eff8}p{line-height:1.6}.wrap{overflow:auto}h1{color:#183e6e}</style>
<h1>Agent Harness · Phase 2 Trace</h1><p>Local runtime events. Fixture runs demonstrate control flow only; they are not model evaluations.</p>"""
    html += f"<p>Started calls: {totals['tool_started']} · Succeeded: {totals['tool_succeeded']} · Failed: {totals['tool_failed']} · Review gates: {totals['review_required']}</p>"
    html += f"<p>Known token subtotal: input {known_input}, output {known_output}. Calls with incomplete usage: {unknown}. Monetary cost is not computed.</p>"
    html += '<div class="wrap"><table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in
        ['Version', 'Stage', 'Event', 'Candidate', 'Attempt', 'Latency ms', 'Input tokens', 'Output tokens', 'Reason'])
    html += '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></html>'
    destination.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, default=Path("harness_trace.html"))
    args = parser.parse_args()
    render_trace(args.trace, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

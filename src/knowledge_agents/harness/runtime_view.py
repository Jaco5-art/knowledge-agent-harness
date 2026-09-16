
"""Render a runtime_report JSON as a local, self-contained HTML report."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import html
import json
from pathlib import Path

def esc(value):
    return html.escape(str(value), quote=True)

def number(value):
    return "???" if value is None else f"{value:,.2f}"

def render(report, source):
    calls = report["calls"]
    groups = defaultdict(list)
    for call in calls:
        label = call.get("agent") or call.get("tool") or "unknown"
        groups[label].append(call)

    ranked = sorted(
        groups.items(),
        key=lambda item: sum(
            c["latency_ms"] for c in item[1]
            if c.get("latency_ms") is not None
        ),
        reverse=True,
    )

    rows = []
    for agent, items in ranked:
        durations = [
            c["latency_ms"] for c in items
            if c.get("latency_ms") is not None
        ]
        inputs = [
            c["input_tokens"] for c in items
            if c.get("input_tokens") is not None
        ]
        outputs = [
            c["output_tokens"] for c in items
            if c.get("output_tokens") is not None
        ]
        rows.append(
            "<tr>"
            f"<td>{esc(agent)}</td>"
            f"<td>{len(items)}</td>"
            f"<td>{sum(c['status']=='tool_failed' for c in items)}</td>"
            f"<td>{number(sum(durations)/1000 if durations else None)}</td>"
            f"<td>{len(durations)}/{len(items)}</td>"
            f"<td>{number(sum(inputs) if inputs else None)}</td>"
            f"<td>{len(inputs)}/{len(items)}</td>"
            f"<td>{number(sum(outputs) if outputs else None)}</td>"
            f"<td>{len(outputs)}/{len(items)}</td>"
            "</tr>"
        )

    failures = Counter(
        (c.get("agent") or c.get("tool") or "unknown",
         c.get("error_type") or "???")
        for c in calls if c["status"] == "tool_failed"
    )
    failure_rows = "".join(
        f"<tr><td>{esc(agent)}</td><td>{esc(error)}</td><td>{count}</td></tr>"
        for (agent, error), count in sorted(failures.items())
    ) or '<tr><td colspan="3">????? tool_failed ??</td></tr>'

    details = "".join(
        "<tr>"
        f"<td>{esc(c.get('agent') or c.get('tool') or 'unknown')}</td>"
        f"<td>{esc(c['status'])}</td>"
        f"<td>{esc(c.get('attempt') if c.get('attempt') is not None else '-')}</td>"
        f"<td>{number(c.get('latency_ms'))}</td>"
        f"<td>{esc(c.get('error_type') or '-')}</td>"
        f"<td>{esc(c['call_id'])}</td>"
        "</tr>"
        for c in calls
    )

    metrics = []
    for field, label in [
        ("input_tokens", "?? token"),
        ("output_tokens", "?? token"),
        ("latency_ms", "?????? ms"),
    ]:
        value = report[field]
        metrics.append(
            f"<p><b>{label}</b>?{number(value['recorded_sum'])}?"
            f"?? {value['recorded_calls']}/{value['total_calls']} ???</p>"
        )

    page = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harness ??????</title>
<style>
body{font:15px/1.65 system-ui,sans-serif;max-width:1250px;
margin:32px auto;padding:0 20px;background:#f4f6fa;color:#19263b}
section{background:white;border:1px solid #dce3ed;border-radius:12px;
padding:22px;margin:18px 0;overflow:auto}
h1{font-size:28px}h2{font-size:20px}
p{color:#46556c}table{border-collapse:collapse;width:100%;white-space:nowrap}
th,td{text-align:left;padding:10px;border-bottom:1px solid #e4e9f0}
th{background:#edf2f9}td{font-variant-numeric:tabular-nums}
</style><body><h1>Harness ??????</h1>
"""
    page += (
        f"<p>???{esc(source.name)}<br>"
        f"???{esc(report['session_id'])}</p>"
        "<section><h2>????</h2>"
        f"<p>??? {report['observed_calls']} ??????"
        f"?? {report['successful_calls']} ??"
        f"?? {report['failed_calls']} ??"
        f"attempt &gt; 1 ??? {report['observed_retry_calls']} ??</p>"
        + "".join(metrics)
        + "<p>???????"
        + str(report["calls_without_terminal_event"])
        + "????????"
        + str(report["terminal_events_without_start"])
        + "?</p></section>"
        "<section><h2>Agent ????</h2><table><thead><tr>"
        "<th>Agent / ??</th><th>??</th><th>??</th>"
        "<th>???</th><th>????</th>"
        "<th>?? token</th><th>????</th>"
        "<th>?? token</th><th>????</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
        "<section><h2>????</h2><table><thead><tr>"
        "<th>Agent / ??</th><th>????</th><th>??</th>"
        "</tr></thead><tbody>"
        + failure_rows
        + "</tbody></table></section>"
        "<section><h2>????</h2>"
        "<p>?????????????????????????"
        "?? token ????????????"
        "??????????????"
        "?????????????????</p></section>"
        "<section><details><summary>??????????????????</summary>"
        "<table><thead><tr><th>Agent / ??</th><th>??</th>"
        "<th>Attempt</th><th>?? ms</th><th>??</th><th>Call ID</th>"
        "</tr></thead><tbody>"
        + details
        + "</tbody></table></details></section></body></html>"
    )
    return page

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.report
    if source is None:
        files = list(Path.cwd().glob("runtime_report_*.json"))
        if not files:
            parser.error("No runtime_report_*.json found; provide --report")
        source = max(files, key=lambda p: p.stat().st_mtime_ns)
    report = json.loads(source.read_text(encoding="utf-8-sig"))
    page = render(report, source)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = args.output or Path(f"runtime_view_{stamp}.html")
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(page)
    print(f"Source: {source.resolve()}")
    print(f"VIEW OK: {destination.resolve()}")
    print("0 API calls.")

if __name__ == "__main__":
    main()

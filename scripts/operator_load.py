"""Replay a CSV against the assistant and estimate operator handoffs (stdlib only)."""
import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def handoff(history, prediction):
    if history >= 3:
        return "history_limit"
    if prediction["intent"] == "human_agent" and prediction["confidence"] > 0.85:
        return "human_agent"
    return None


def clarification_signal(answer):
    if answer.get("type") == "clarification" or answer.get("follow_up_question"):
        return "structured"
    if "?" in str(answer.get("content", "")):
        return "question_mark_heuristic"
    return "none"


def post(base, endpoint, payload, timeout):
    request = Request(base.rstrip("/") + "/assistant/" + endpoint,
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"{endpoint}: HTTP {exc.code}: {exc.read(2000).decode(errors='replace')}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"{endpoint}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"{endpoint}: expected JSON object")
    return result, round((time.perf_counter() - started) * 1000, 3)


def load_rows(args):
    with args.dataset.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=args.delimiter)
        fields = reader.fieldnames or []
        required = [args.text_column]
        required += [c for c in (args.session_column, args.history_column) if c]
        if any(c not in fields for c in required):
            raise ValueError(f"Required columns: {required}; found: {fields}. Knowledge CSV is not a request dataset.")
        rows = []
        histories = Counter()
        for number, row in enumerate(reader, 2):
            text = row[args.text_column].strip()
            if not text:
                raise ValueError(f"Line {number}: empty text")
            session = row[args.session_column].strip() if args.session_column else str(number)
            if not session:
                raise ValueError(f"Line {number}: empty session ID")
            history = int(row[args.history_column]) if args.history_column else (
                histories[session] if args.session_column else args.default_history)
            if history < 0:
                raise ValueError(f"Line {number}: history must be >= 0")
            histories[session] += 1
            rows.append({"line": number, "session": session, "history": history,
                         "text": text, "expected_intent": row.get(args.intent_column, "")})
    if not rows:
        raise ValueError("Empty dataset")
    # Sample whole sessions; never remove turns from a selected session.
    if args.sample_sessions:
        ids = list(dict.fromkeys(r["session"] for r in rows))
        selected = set(random.Random(args.seed).sample(ids, min(args.sample_sessions, len(ids))))
        rows = [r for r in rows if r["session"] in selected]
    return rows


def replay(args, rows, send=post):
    with args.knowledge.open(encoding="utf-8-sig", newline="") as stream:
        knowledge = {r["Intent"] for r in csv.DictReader(stream)}
    transferred, uncertain = set(), set()
    results = []
    for row in rows:
        record = dict(row, decision="", reason="", prediction=None, generation=None, error="")
        session = row["session"]
        try:
            if session in transferred:
                record.update(decision="operator", reason="already_transferred")
            elif row["history"] >= 3:
                record.update(decision="handoff", reason="history_limit")
                transferred.add(session)
            elif session in uncertain:
                record.update(decision="unknown", reason="previous_classification_error")
            else:
                text = row["text"]
                if args.preprocess:
                    processed, latency = send(args.base_url, "process", {"text": text}, args.timeout)
                    text = processed["processed"]
                    record["process_ms"] = latency
                pred, latency = send(args.base_url, "classify", {"text": text}, args.timeout)
                confidence = pred.get("confidence")
                if (not isinstance(pred.get("intent"), str) or isinstance(confidence, bool)
                        or not isinstance(confidence, (float, int)) or not math.isfinite(confidence)
                        or not 0 <= confidence <= 1):
                    raise RuntimeError("classify: invalid intent/confidence")
                record.update(prediction=pred, classify_ms=latency,
                              knowledge_covered=pred["intent"] in knowledge)
                reason = handoff(row["history"], pred)
                if reason:
                    record.update(decision="handoff", reason=reason)
                    transferred.add(session)
                else:
                    record["decision"] = "bot"
                    if not args.skip_generation:
                        payload = {"text": text}
                        if args.order_context:
                            payload["order_context"] = args.order_context
                        generated, latency = send(args.base_url, "generate-answer", payload, args.timeout)
                        record.update(generation=generated, generate_ms=latency,
                                      clarification_signal=clarification_signal(generated))
        except (RuntimeError, KeyError, TypeError) as exc:
            record["error"] = str(exc)
            if not record["decision"]:
                record["decision"] = "unknown"
                uncertain.add(session)
        results.append(record)
        # Flush every row so a slow or interrupted run retains completed requests.
        args.results_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        args.results_stream.flush()
    sessions = {r["session"] for r in rows}
    unresolved = uncertain - transferred
    decisions = Counter(r["decision"] for r in results)
    reasons = Counter(r["reason"] for r in results if r["decision"] == "handoff")
    total = len(sessions)
    lower, upper = len(transferred) / total, (len(transferred) + len(unresolved)) / total
    return {
        "sessions": total, "messages": len(rows), "decisions": dict(decisions),
        "handoff_reasons": dict(reasons), "transferred_sessions": len(transferred),
        "unresolved_sessions": len(unresolved),
        "operator_session_share": lower if not unresolved else None,
        "operator_session_share_bounds": [lower, upper],
        "operator_messages": decisions["handoff"] + decisions["operator"],
        "api_errors": sum(bool(r["error"]) for r in results),
        "generation_successes": sum(r["generation"] is not None for r in results),
        "clarification_signals": dict(Counter(r["clarification_signal"] for r in results
                                             if r.get("clarification_signal") is not None)),
        "classification_by_expected_intent": {
            intent: {
                "classified": sum(r["expected_intent"] == intent and r["prediction"] is not None for r in results),
                "correct": sum(r["expected_intent"] == intent and r["prediction"] is not None
                               and r["prediction"]["intent"] == intent for r in results),
                "handoffs": sum(r["expected_intent"] == intent and r["decision"] == "handoff" for r in results),
            } for intent in sorted({r["expected_intent"] for r in results if r["expected_intent"]})
        },
        "forecast": None if args.sessions_per_day is None else {
            "operator_sessions_per_day_bounds": [args.sessions_per_day * lower, args.sessions_per_day * upper],
            "operator_hours_per_day_bounds": None if args.handle_minutes is None else [
                args.sessions_per_day * share * args.handle_minutes / 60 for share in (lower, upper)]},
        "assumptions": {
            "session_order": "CSV order within each session; no sorting",
            "history": "explicit column" if args.history_column else (
                "preceding CSV rows in session (one row = one counted message)" if args.session_column
                else f"independent requests, fixed previous history={args.default_history}"),
            "generation_enabled": not args.skip_generation,
            "preprocessing_enabled": args.preprocess,
            "handoff_is_terminal": True,
            "errors_are_not_handoffs": True,
            "knowledge_is_not_traffic_distribution": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--knowledge", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/knowledge_db/knowledge_db_v1.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--intent-column", default="intent")
    parser.add_argument("--session-column")
    parser.add_argument("--history-column")
    parser.add_argument("--default-history", type=int, default=0)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--sample-sessions", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--order-context")
    parser.add_argument("--sessions-per-day", type=float)
    parser.add_argument("--handle-minutes", type=float)
    parser.add_argument("--output", type=Path, default=Path("operator-load-results"))
    args = parser.parse_args()
    if (args.default_history < 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or len(args.delimiter) != 1
            or (args.sample_sessions is not None and args.sample_sessions <= 0)
            or any(v is not None and (not math.isfinite(v) or v < 0)
                   for v in (args.sessions_per_day, args.handle_minutes))):
        parser.error("Invalid numeric option or delimiter")
    try:
        rows = load_rows(args)
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "requests.jsonl").open("w", encoding="utf-8") as stream:
            args.results_stream = stream
            summary = replay(args, rows)
        summary["configuration"] = {k: str(v) if isinstance(v, Path) else v
                                    for k, v in vars(args).items() if k != "results_stream"}
        rendered = json.dumps(summary, ensure_ascii=False, indent=2)
        (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
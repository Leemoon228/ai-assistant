"""Generate and optionally replay explicit synthetic scenarios from the knowledge CSV."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from operator_load import replay


def rough_model(summary, args):
    """Initial human routing plus three iid bot followups; not a measured session rate."""
    total = summary["sessions"]
    human = summary["handoff_reasons"].get("human_agent", 0)
    unknown = summary["unresolved_sessions"]
    successes = summary["generation_successes"]
    signals = summary["clarification_signals"]
    count = signals.get("structured", 0)
    if args.clarification_mode == "question-mark":
        count += signals.get("question_mark_heuristic", 0)
    observed = count / successes if successes else None
    p = args.followup_probability if args.followup_probability is not None else observed
    missing_bot_generations = summary["decisions"].get("bot", 0) - successes
    complete = not unknown and (args.followup_probability is not None or missing_bot_generations == 0)
    # If every request is handed off immediately, p is immaterial.
    if human == total:
        share = 1.0
    else:
        share = human / total + (1 - human / total) * p ** 3 if p is not None and complete else None
    return {
        "model": "h + (1-h) * p^3",
        "p_source": "manual assumption" if args.followup_probability is not None else args.clarification_mode,
        "generated_answers": successes, "answers_with_followup_signal": count,
        "observed_followup_signal_share": observed,
        "p_used": p, "history_handoff_probability_given_bot": None if p is None else p ** 3,
        "initial_human_handoff_share": human / total if not unknown else None,
        "missing_bot_generations": missing_bot_generations,
        "estimated_total_operator_share": share,
        "operator_sessions_per_day": None if share is None or args.sessions_per_day is None else share * args.sessions_per_day,
        "operator_hours_per_day": None if share is None or args.sessions_per_day is None or args.handle_minutes is None
        else share * args.sessions_per_day * args.handle_minutes / 60,
        "assumptions": ["one initial utterance per synthetic session; equal intent weights",
                        "p is identical and independent across three consecutive bot turns",
                        "every bot followup leads to another customer message",
                        "human_agent routing is accounted for on the first turn only",
                        "no later unsolicited customer followups or other escalation rules",
                        "question marks are a noisy proxy, not a quality or resolution judgment"],
    }


# Handwritten customer utterances: Knowledge contains answers, not user questions.
QUESTIONS = {
    "close_account": "How do I delete my account?",
    "recover_password": "I forgot my password. How can I reset it?",
    "order_history": "Where can I see my previous orders?",
    "open_account": "How can I create an account?",
    "change_account": "How can I update my account details?",
    "use_app": "How do I use your mobile app?",
    "technical_issue": "Your website keeps crashing when I try to use it.",
    "remove_product": "How do I remove an item from my shopping cart?",
    "add_product": "How do I add an item to my cart?",
    "human_agent": "Please connect me to a human agent.",
    "customer_service": "How can I contact customer service?",
    "wrong_item": "I received a different item from the one I ordered.",
    "delivery_issue": "There is a problem with my delivery. Can you help?",
    "track_delivery": "How can I track my delivery?",
    "damaged_delivery": "My package arrived damaged. What should I do?",
    "shipping_costs": "How much does shipping cost?",
    "delivery_time": "How long does delivery take?",
    "missing_item": "An item is missing from my package.",
    "submit_product_idea": "How can I suggest an idea for a new product?",
    "submit_product_feedback": "I would like to leave feedback about a product.",
    "submit_feedback": "How can I send feedback about your service?",
    "request_invoice": "Can I get an invoice for my order?",
    "cancel_order": "I want to cancel my order.",
    "track_order": "Where is my order? How can I track its status?",
    "change_order": "Can I change an order I have already placed?",
    "payment_issue": "My payment failed. What should I do?",
    "payment_methods": "Which payment methods do you accept?",
    "pay": "How do I pay for my order?",
    "availability_online": "Is this product available to buy online?",
    "product_issue": "The product I bought does not work properly.",
    "exchange_product_in_store": "Can I exchange my product at a physical store?",
    "exchange_product": "How can I exchange this product for another one?",
    "product_information": "Where can I find the product specifications?",
    "availability": "Is this product in stock?",
    "availability_in_store": "Is this item available in your physical store?",
    "refund_status": "What is the status of my refund?",
    "return_product_in_store": "How can I return an item at your store?",
    "return_policy": "What is your return policy?",
    "refund_policy": "What are the conditions for getting a refund?",
    "return_product": "How can I return a product?",
    "return_product_online": "How can I return a product I purchased online?",
    "request_refund": "I would like to request a refund.",
    "sales_period": "When is your next sale?",
    "store_opening_hours": "What time does your store open and close?",
    "store_location": "Where is your nearest store?",
    "request_right_to_rectification": "How can I exercise my right to correct my personal data?",
}


def generate(knowledge, folder, lengths, variants):
    with knowledge.open(encoding="utf-8-sig", newline="") as stream:
        articles = list(csv.DictReader(stream))
    if not articles or any(not row.get("Intent") or not row.get("Knowledge") for row in articles):
        raise ValueError("Knowledge CSV must contain nonempty Intent and Knowledge")
    intents = [row["Intent"] for row in articles]
    if len(set(intents)) != len(intents):
        raise ValueError("Duplicate intents in knowledge CSV")
    missing = set(intents) - QUESTIONS.keys()
    if missing:
        raise ValueError(f"Add customer utterances for these intents: {sorted(missing)}")
    scenarios = {}
    for length in lengths:
        rows = []
        for intent in intents:
            question = QUESTIONS[intent]
            prompts = [question, "Hello, I need some help. " + question,
                       question + " Could you help me with this?"][:variants]
            for variant, text in enumerate(prompts):
                session = f"{intent}-{variant}"
                for history in range(length):
                    # Intent stays visible on later turns; no invented generation-dependent outcome.
                    message = text if history == 0 else "I still need help with this request. " + text
                    rows.append(dict(line=len(rows) + 2, session=session, history=history,
                                     text=message, expected_intent=intent))
        path = folder / f"requests-{length}-turns.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["session_id", "text", "history_count", "intent"])
            writer.writeheader()
            writer.writerows(dict(session_id=r["session"], text=r["text"], history_count=r["history"],
                                  intent=r["expected_intent"]) for r in rows)
        scenarios[length] = rows
    return scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/knowledge_db/knowledge_db_v1.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--variants", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--run", action="store_true", help="Call the live service after generating CSVs")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--sessions-per-day", type=float)
    parser.add_argument("--handle-minutes", type=float)
    parser.add_argument("--followup-probability", type=float, help="Manually assumed p instead of answer-derived estimate")
    parser.add_argument("--clarification-mode", choices=["structured", "question-mark"], default="question-mark")
    args = parser.parse_args()
    if args.followup_probability is not None and not 0 <= args.followup_probability <= 1:
        parser.error("Followup probability must be between 0 and 1")
    if any(length < 1 for length in args.lengths) or len(set(args.lengths)) != len(args.lengths):
        parser.error("Lengths must be distinct positive integers")
    import math
    if not math.isfinite(args.timeout) or args.timeout <= 0 or any(
        value is not None and (not math.isfinite(value) or value < 0)
        for value in (args.sessions_per_day, args.handle_minutes)
    ):
        parser.error("Invalid numeric option")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        scenarios = generate(args.knowledge, args.output, args.lengths, args.variants)
        report = {"synthetic": True, "assumptions": {
            "intent_distribution": "uniform over knowledge intents; not observed traffic",
            "utterances": "handwritten English templates, not real customer requests",
            "history": "one prior user CSV row = one counted history message; excludes bot replies",
            "lengths": "separate fixed-length counterfactual scenarios, not observed session lengths",
            "followups": "scripted repetition, independent of generated answer quality",
            "no_aggregate_share": "scenario weights are unknown; do not average as a production estimate",
        }, "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "scenarios": {}}
        for length, rows in scenarios.items():
            if args.run:
                config = SimpleNamespace(**vars(args), history_column="history_count",
                                         session_column="session_id", default_history=0, order_context=None)
                with (args.output / f"results-{length}-turns.jsonl").open("w", encoding="utf-8") as stream:
                    config.results_stream = stream
                    summary = replay(config, rows)
            else:
                summary = {"sessions": len({r["session"] for r in rows}), "messages": len(rows),
                           "status": "generated only, service not called"}
            report["scenarios"][str(length)] = summary
            if args.run and length == 1:
                report["rough_probability_model"] = rough_model(summary, args)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        (args.output / "scenarios.json").write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
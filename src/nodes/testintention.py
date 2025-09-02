import os
import sys
import json
from pprint import pprint

# Adjust import to your structure: src/helper/normalizer.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.helpers.intentnormalizer import IntentPolicyNormalizer  # noqa: E402


def case_disk_cleanup():
    """Alert with high % and a ticket → disk_cleanup."""
    state = {
        "classification": {"classification": "Alert", "confidence": 96},
        "entities": {
            "servers": [{"value": "db-prod-03", "confidence": 95}],
            "tickets": [{"value": "INC112233", "confidence": 95}],
            "requests": [],
            "alerts": [{"value": "Disk utilization exceeded 96%", "confidence": 90}],
            "events": [{"value": "Disk utilization exceeded threshold at 96%", "confidence": 90}],
            "links": [{"server": "db-prod-03", "ticket": "INC112233"}],
        },
    }
    # Pretend LLM guessed wrong
    decision = {"intent": "ticket_update", "action": None, "route": None, "confidence": 60, "reasons": []}
    expect_intent = "disk_cleanup"
    expect_route = "disk_alert_subgraph"
    return state, decision, expect_intent, expect_route


def case_fs_extension():
    """Request to extend filesystem → fs_extension."""
    state = {
        "classification": {"classification": "Request", "confidence": 93},
        "entities": {
            "servers": [{"value": "app-finance-01", "confidence": 95}],
            "tickets": [{"value": "INC445566", "confidence": 95}],
            "requests": [{"value": "Filesystem Extension", "confidence": 90}],
            "alerts": [],
            "events": [],
            "links": [{"server": "app-finance-01", "ticket": "INC445566"}],
        },
    }
    decision = {"intent": "unknown", "action": None, "route": None, "confidence": 70, "reasons": []}
    expect_intent = "fs_extension"
    expect_route = "fs_extension_subgraph"
    return state, decision, expect_intent, expect_route


def case_ticket_update():
    """Status update email with a ticket, no strong operational signal → ticket_update."""
    state = {
        "classification": {"classification": "Alert", "confidence": 90},
        "entities": {
            "servers": [{"value": "linux2", "confidence": 95}],
            "tickets": [{"value": "INC776655", "confidence": 95}],
            "requests": [],
            "alerts": [],
            "events": [{"value": "Disk utilization reduced to 55%", "confidence": 85}],
            "links": [{"server": "linux2", "ticket": "INC776655"}],
        },
    }
    decision = {"intent": "unknown", "action": None, "route": None, "confidence": 60, "reasons": []}
    expect_intent = "ticket_update"
    expect_route = "ticket_update_subgraph"
    return state, decision, expect_intent, expect_route


def run_case(normalizer: IntentPolicyNormalizer, name: str, state, decision, expect_intent, expect_route) -> bool:
    print(f"\n=== CASE: {name} ===")
    print("Input decision (LLM guess):", decision)
    final = normalizer.normalize_decision(state, decision)
    # Optional: bump to simulate your node’s confidence policy
    if final.get("intent") and final.get("action") == "route_subgraph":
        final["confidence"] = max(final.get("confidence", 0), 90)

    print("Final normalized decision:")
    pprint(final)

    intent_ok = final.get("intent") == expect_intent
    route_ok = final.get("route") == expect_route
    if not (intent_ok and route_ok):
        print(f"❌ FAILED: expected intent={expect_intent}, route={expect_route}")
        return False

    print("✅ PASS")
    # Short trace peek (if your normalizer attaches debug)
    dbg = (final.get("debug") or {}).get("normalizer_trace")
    if dbg:
        print("Trace (first 2 entries):")
        pprint(dbg[:2])
    return True


def main():
    # Config JSON is expected next to this file: src/nodes/intent_config.json
    config_path = os.path.join(os.path.dirname(__file__), "intent_config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: intent_config.json not found at {config_path}", file=sys.stderr)
        sys.exit(2)

    normalizer = IntentPolicyNormalizer(config_path=config_path)

    # Smoke-check config loaded as expected
    print("Loaded intents:", list(normalizer.config.keys()))
    print("disk_cleanup rule route:", (normalizer.config.get("disk_cleanup") or {}).get("route"))

    cases = [
        ("disk_cleanup",) + case_disk_cleanup(),
        ("fs_extension",) + case_fs_extension(),
        ("ticket_update",) + case_ticket_update(),
    ]

    failures = 0
    for name, state, decision, expect_intent, expect_route in cases:
        ok = run_case(normalizer, name, state, decision, expect_intent, expect_route)
        failures += 0 if ok else 1

    if failures:
        print(f"\nSUMMARY: ❌ {failures} case(s) failed.")
        sys.exit(1)
    else:
        print("\nSUMMARY: ✅ All cases passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
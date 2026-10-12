"""
Offline test for call_agent_for_json (the retry on malformed agent JSON).
No sign-in, no model calls: a fake response object stands in for the agent.

Run: python test_json_retry.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

from foundry_clients import call_agent_for_json

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


class Reply:
    def __init__(self, text):
        self.output_text = text


def scripted(*texts):
    calls = []

    def make_call():
        calls.append(1)
        return Reply(texts[min(len(calls), len(texts)) - 1])
    return make_call, calls


GOOD = '{"scenarios": []}'
BAD = '{"scenarios": [ {"source": "a "quoted" word"} ]}'

fn, calls = scripted(GOOD)
check("valid JSON on first try: one call", call_agent_for_json("t", fn) == {"scenarios": []} and len(calls) == 1)

fn, calls = scripted("```json\n" + GOOD + "\n```")
check("code fence is stripped, one call", call_agent_for_json("t", fn) == {"scenarios": []} and len(calls) == 1)

fn, calls = scripted(BAD, GOOD)
check("malformed then valid: retried once and succeeds", call_agent_for_json("t", fn) == {"scenarios": []} and len(calls) == 2, len(calls))

fn, calls = scripted(BAD, BAD, GOOD)
try:
    call_agent_for_json("t", fn)
    raised = False
except RuntimeError as e:
    raised = "after 2 attempts" in str(e)
check("malformed twice: raises after exactly 2 calls, no third", raised and len(calls) == 2, len(calls))

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_failure_raw_t.txt")
check("failed reply saved under the agent's own file name", os.path.exists(path) and open(path, encoding="utf-8").read() == BAD)
if os.path.exists(path):
    os.remove(path)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)

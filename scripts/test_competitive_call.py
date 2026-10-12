"""
Offline test for how run_competitive_analysis chooses between the v3 JSON reading and the v4 text layout, and for the
one retry on an unreadable v4 reply. A fake client stands in for Azure; the source check's page fetch is replaced so
nothing touches the network.

Run: python test_competitive_call.py
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("FOUNDRY_PROJECT_ENDPOINT", "https://example.invalid")
os.environ.setdefault("COMPETITIVE_ANALYSIS_AGENT_VERSION", "3")

import run_competitive_analysis as rca

HERE = os.path.dirname(os.path.abspath(__file__))
failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {str(detail)[:300]}"))
    if not condition:
        failures.append(name)


real = json.load(open(os.path.join(HERE, "fixtures", "competitive_v4_billing.json"), encoding="utf-8"))


class Resp:
    def __init__(self, d, text=None):
        self.d, self.output_text = d, text if text is not None else ""

    def model_dump(self):
        return self.d


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

        outer = self

        class _R:
            def create(self, **kw):
                outer.calls.append(kw)
                return outer.replies.pop(0)
        self.responses = _R()


# make the budget wrapper and the page fetch inert
rca.budgeted_call = lambda fn, *a, **k: fn()
rca.check_competitor_sources = lambda output: output

orig = rca.AGENT_VERSION
try:
    rca.AGENT_VERSION = "3"
    check("version 3 is read as JSON", rca.uses_text_layout() is False)
    rca.AGENT_VERSION = "4"
    check("version 4 is read as the text layout", rca.uses_text_layout() is True)
    rca.AGENT_VERSION = "12"
    check("later versions use the text layout too", rca.uses_text_layout() is True)
    rca.AGENT_VERSION = "latest"
    check("a version that isn't a number falls back to JSON rather than crashing", rca.uses_text_layout() is False)

    rca.AGENT_VERSION = "4"
    client = FakeClient([Resp(real)])
    _, out = rca.call_competitive_analysis("a billing feature", openai_client=client)
    check("a v4 reply is parsed into the v3 structure", [c["product"] for c in out["competitors"]] == ["Maven AGI", "Ada", "Fin AI Agent"], out)
    check("the call names the agent and version 4", client.calls[0]["extra_body"]["agent_reference"] == {"name": "competitive-analysis", "version": "4", "type": "agent_reference"}, client.calls[0])
    check("a readable reply costs one call", len(client.calls) == 1)

    bad = Resp({"output": [{"type": "message", "content": [{"type": "output_text", "text": "Some free-form chat answer.", "annotations": []}]}]}, text="Some free-form chat answer.")
    client = FakeClient([bad, Resp(real)])
    _, out = rca.call_competitive_analysis("a billing feature", openai_client=client)
    check("an unreadable reply is asked for once more, and the second is used", len(client.calls) == 2 and len(out["competitors"]) == 3)

    client = FakeClient([bad, bad])
    try:
        rca.call_competitive_analysis("a billing feature", openai_client=client)
        raised = False
    except RuntimeError as e:
        raised = "did not follow the v4 layout" in str(e)
    check("two unreadable replies in a row raise a plain error (the Orchestrator then marks the agent failed), after exactly two calls", raised and len(client.calls) == 2)

    # v3 path unchanged
    rca.AGENT_VERSION = "3"
    v3_json = json.dumps({"competitors": [{"product": "X"}], "grounding_status": "grounded", "missing_inputs": []})
    client = FakeClient([Resp({}, text=v3_json)])
    _, out = rca.call_competitive_analysis("a billing feature", openai_client=client)
    check("the v3 JSON path still works", out["competitors"][0]["product"] == "X", out)
finally:
    rca.AGENT_VERSION = orig

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)

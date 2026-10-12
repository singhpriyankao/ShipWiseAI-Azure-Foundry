"""
Offline tests for feature_requirements.py (the step that reads a feature description and works out the
usage, reply style and caveats to build the Model Stack options for). A fake model stands in for
gpt-4.1-nano; no network, no sign-in.

Run: python test_feature_requirements.py
"""
import json
import warnings
warnings.filterwarnings("ignore")

import feature_requirements as fr

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


DESC = "A chatbot on our help page that answers billing questions from our knowledge base. We get about 500 chats a day."
GOOD = {"stated_volume": {"quote": "about 500 chats a day", "amount": 500, "period": "day"},
        "avg_input_tokens": 2000, "avg_output_tokens": 250, "reply_style": "streamed_chat", "feature_category": "chat_assistant",
        "needs": {"image_input": False, "audio_input": False, "very_long_documents": False, "open_source_required": False, "data_residency_or_on_prem": False},
        "reason": "A knowledge-base chatbot sends the conversation plus retrieved articles."}

ok = fr.validate_interpretation(GOOD, DESC)
check("a good interpretation is used", ok["source"] == "model" and ok["usage"]["avg_input_tokens"] == 2000 and ok["usage"]["avg_output_tokens"] == 250, ok)
check("a stated volume is converted to requests a month (500 a day = 15,000)", ok["usage"]["requests_per_month"] == 15000 and ok["usage"]["volume_basis"] == "stated", ok["usage"])
check("token sizes are labelled estimated, never stated", ok["usage"]["token_basis"] == "estimated")
check("the reply style is taken", ok["reply_style"] == "streamed_chat")

# ---- the volume must be in the PM's own words
no_vol = dict(GOOD, stated_volume=None)
r = fr.validate_interpretation(no_vol, "A chatbot for billing questions.")
check("no stated volume -> the default volume, labelled assumed", r["usage"]["requests_per_month"] == fr.DEFAULT_USAGE["requests_per_month"] and r["usage"]["volume_basis"] == "assumed" and r["usage"]["basis"] == "assumed", r["usage"])
check("...and the reason says the volume was assumed", "was not given" in r["usage"]["reason"], r["usage"]["reason"])
invented = dict(GOOD, stated_volume={"quote": "about 9,000 chats a day", "amount": 9000, "period": "day"})
r = fr.validate_interpretation(invented, DESC)
check("a quote that isn't in the description is rejected, so a guessed volume can't pass as stated", r["usage"]["volume_basis"] == "assumed" and any("could not be confirmed" in p for p in r["problems"]), r)
wrong_amount = dict(GOOD, stated_volume={"quote": "about 500 chats a day", "amount": 5000, "period": "day"})
check("an amount that isn't in the quote is rejected", fr.validate_interpretation(wrong_amount, DESC)["usage"]["volume_basis"] == "assumed")
check("a quote with different capitalisation and spacing still verifies", fr.verify_stated_volume({"quote": "About  500 CHATS a day", "amount": 500, "period": "day"}, DESC) == 15000)
check("'20k a month' counts as 20,000", fr.verify_stated_volume({"quote": "20k tickets a month", "amount": 20000, "period": "month"}, "We see 20k tickets a month.") == 20000)
check("a weekly figure converts (1,200 a week is about 5,200 a month)", fr.verify_stated_volume({"quote": "1,200 tickets a week", "amount": 1200, "period": "week"}, "About 1,200 tickets a week.") == 5200)
check("a yearly figure converts", fr.verify_stated_volume({"quote": "120,000 chats a year", "amount": 120000, "period": "year"}, "120,000 chats a year") == 10000)
check("an unknown period is rejected", fr.verify_stated_volume({"quote": "500 a fortnight", "amount": 500, "period": "fortnight"}, "500 a fortnight") is None)
check("a boolean or text amount is rejected", fr.verify_stated_volume({"quote": "500 a day", "amount": True, "period": "day"}, "500 a day") is None and fr.verify_stated_volume({"quote": "500 a day", "amount": "lots", "period": "day"}, "500 a day") is None)
check("an absurd volume is rejected", fr.verify_stated_volume({"quote": "900,000,000,000 a day", "amount": 900_000_000_000, "period": "day"}, "900,000,000,000 a day") is None)
check("a non-dict stated volume is rejected without crashing", fr.verify_stated_volume("500 a day", DESC) is None and fr.verify_stated_volume(None, DESC) is None)

# ---- token sizes
bad_tokens = dict(GOOD, avg_input_tokens=-5)
r = fr.validate_interpretation(bad_tokens, DESC)
check("token sizes out of range -> defaults, and the problem is reported", r["usage"]["avg_input_tokens"] == fr.DEFAULT_USAGE["avg_input_tokens"] and r["source"] == "default" and any("token sizes" in p for p in r["problems"]), r)
check("a text token size is rejected", fr.validate_interpretation(dict(GOOD, avg_output_tokens="a few"), DESC)["source"] == "default")
check("a huge token size is rejected", fr.validate_interpretation(dict(GOOD, avg_input_tokens=10_000_000), DESC)["source"] == "default")
check("a fractional token size is rounded", fr.validate_interpretation(dict(GOOD, avg_input_tokens=1999.6), DESC)["usage"]["avg_input_tokens"] == 2000)

# ---- reply style
check("single_response is accepted for a feature that is not a chat", fr.validate_interpretation(dict(GOOD, feature_category="classification_tagging", reply_style="single_response"), DESC)["reply_style"] == "single_response")
r = fr.validate_interpretation(dict(GOOD, feature_category="chat_assistant", reply_style="single_response"), DESC)
check("a chat assistant read as single_response is corrected to a streamed chat, and the correction is reported",
      r["reply_style"] == "streamed_chat" and any("treated as a streamed chat" in p for p in r["problems"]), r)
check("a chat assistant already read as a streamed chat adds no problem", ok["reply_style"] == "streamed_chat" and not any("streamed chat" in p for p in ok["problems"]), ok["problems"])
check("the correction does not touch other categories (a summarizer stays single_response)",
      fr.validate_interpretation(dict(GOOD, feature_category="summarization", reply_style="single_response"), DESC)["reply_style"] == "single_response")
r = fr.validate_interpretation(dict(GOOD, reply_style="sometimes"), DESC)
check("an unknown reply style falls back to a streamed chat and says so", r["reply_style"] == "streamed_chat" and any("reply style" in p for p in r["problems"]))

# ---- token sizes outside the typical range for the kind of feature
r = fr.validate_interpretation(dict(GOOD, avg_input_tokens=200, avg_output_tokens=250), "A billing chatbot for our help page.")
check("a chat estimated at 200 tokens in is moved to the bottom of the chat range (1,500), output left alone",
      r["usage"]["avg_input_tokens"] == 1500 and r["usage"]["avg_output_tokens"] == 250 and any("moved to the nearest end" in p for p in r["problems"]), r["usage"])
check("...and the reason says the estimate was moved", "nearest end of that range" in r["usage"]["reason"], r["usage"]["reason"])
r = fr.validate_interpretation(dict(GOOD, avg_input_tokens=9000), "A billing chatbot for our help page.")
check("a chat estimated above the range is moved down to the top (3,000)", r["usage"]["avg_input_tokens"] == 3000, r["usage"])
r = fr.validate_interpretation(dict(GOOD, feature_category="routing_triage", reply_style="single_response", avg_input_tokens=50, avg_output_tokens=10), "Route each ticket to a team queue.")
check("a router at 50 tokens in is moved to 300 (routing uses the tagging range); 10 out is already in range", r["usage"]["avg_input_tokens"] == 300 and r["usage"]["avg_output_tokens"] == 10, r["usage"])
r = fr.validate_interpretation(dict(GOOD, avg_input_tokens=100, avg_output_tokens=50), "A chatbot where each customer message is about 100 tokens and our answers are about 50 tokens.")
check("a size the PM stated in tokens is never overridden", r["usage"]["avg_input_tokens"] == 100 and r["usage"]["avg_output_tokens"] == 50 and not any("nearest end" in p for p in r["problems"]), r["usage"])
r = fr.validate_interpretation(dict(GOOD, avg_input_tokens=200), "A chatbot; our customers write about 150 words per message.")
check("a size stated in words is not overridden either", r["usage"]["avg_input_tokens"] == 200, r["usage"])
r = fr.validate_interpretation(dict(GOOD, avg_input_tokens=200), "A billing chatbot, 90% accuracy, under 800ms, $40K a year.")
check("other numbers (accuracy, latency, budget) are not mistaken for a stated size", r["usage"]["avg_input_tokens"] == 1500, r["usage"])
r = fr.validate_interpretation(dict(GOOD, feature_category="other", avg_input_tokens=30), "Something unusual.")
check("category 'other' has no range, so nothing is moved", r["usage"]["avg_input_tokens"] == 30 and not any("nearest end" in p for p in r["problems"]), r["usage"])
r = fr.validate_interpretation(dict(GOOD, avg_input_tokens=-5), "A billing chatbot.")
check("token sizes that were already rejected (defaults) are not then moved", r["source"] == "default" and r["usage"]["avg_input_tokens"] == fr.DEFAULT_USAGE["avg_input_tokens"], r["usage"])
check("a size inside the range is left alone and reports no problem", ok["usage"]["avg_input_tokens"] == 2000 and not any("nearest end" in p for p in ok["problems"]), ok["problems"])

# ---- caveats
needs = dict(GOOD, needs={"image_input": True, "audio_input": False, "very_long_documents": False, "open_source_required": True, "data_residency_or_on_prem": False})
r = fr.validate_interpretation(needs, DESC)
check("requirements we can't check become warnings", len(r["caveats"]) == 2 and any("images" in c for c in r["caveats"]) and any("open-source" in c for c in r["caveats"]), r["caveats"])
check("no needs, no caveats", ok["caveats"] == [])
check("only a literal true triggers a caveat (not a string)", fr.validate_interpretation(dict(GOOD, needs={"image_input": "yes"}), DESC)["caveats"] == [])

# ---- the kind of feature
check("the feature category is taken", ok["feature_category"] == "chat_assistant", ok.get("feature_category"))
r = fr.validate_interpretation(dict(GOOD, feature_category="spaceship"), DESC)
check("an unknown category is recorded as 'other' and reported", r["feature_category"] == "other" and any("feature category" in p for p in r["problems"]), r)
check("a missing category is recorded as 'other'", fr.validate_interpretation({k: v for k, v in GOOD.items() if k != "feature_category"}, DESC)["feature_category"] == "other")
check("garbage in gives the category 'other'", fr.validate_interpretation(None, DESC)["feature_category"] == "other")
check("every category in the prompt is a valid category", all(c in fr.INTERPRET_PROMPT for c in fr.FEATURE_CATEGORIES))

# ---- garbage in
for bad in (None, "text", [], 5):
    r = fr.validate_interpretation(bad, DESC)
    check(f"garbage ({bad!r}) gives the defaults, not an error", r["source"] == "default" and r["usage"]["requests_per_month"] == fr.DEFAULT_USAGE["requests_per_month"])
check("the defaults themselves are untouched by a run", fr.DEFAULT_USAGE["requests_per_month"] == 20000 and fr.DEFAULT_USAGE["volume_basis"] == "assumed")

# ---- the whole step, with a fake model
class Fake:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.last = reply, error, None
        outer = self

        class _C:
            def create(self, **kw):
                outer.last = kw
                if outer.error:
                    raise outer.error
                msg = type("M", (), {"content": outer.reply if isinstance(outer.reply, str) else json.dumps(outer.reply)})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})
        self.chat = type("Chat", (), {"completions": _C()})


f = Fake(GOOD)
r = fr.interpret_feature(f, DESC)
check("interpret: one call, JSON mode, on the small model", f.last["model"] == "gpt-4.1-nano" and f.last["response_format"] == {"type": "json_object"})
check("interpret: the description reaches the model inside tags, as data", "<description>" in f.last["messages"][1]["content"] and DESC in f.last["messages"][1]["content"])
check("interpret: a good reply is used", r["source"] == "model" and r["usage"]["requests_per_month"] == 15000)
r = fr.interpret_feature(Fake(error=RuntimeError("down")), DESC)
check("interpret: a failed call gives the defaults, never an error", r["source"] == "default" and any("call failed" in p for p in r["problems"]), r)
r = fr.interpret_feature(Fake("this is not json"), DESC)
check("interpret: a non-JSON reply gives the defaults, never an error", r["source"] == "default" and r["usage"]["requests_per_month"] == 20000, r)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)

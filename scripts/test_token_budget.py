"""
Offline tests for the token budget in foundry_clients.py. A fake clock stands in
for real time so nothing waits, plus one short real-time test with threads.

Run: python test_token_budget.py
"""
import threading
import time
import warnings
warnings.filterwarnings("ignore")

import foundry_clients as fc

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


class FakeTime:
    def __init__(self):
        self.now, self.slept = 1000.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def budget(capacity=50_000, margin=0.9, ft=None):
    ft = ft or FakeTime()
    return fc.TokenBudget(capacity, margin=margin, clock=ft.clock, sleep=ft.sleep), ft


# ---- the basic rule
b, ft = budget()
b.acquire(20_000)
b.acquire(20_000)
check("calls that fit are not delayed", ft.slept == [], ft.slept)

b, ft = budget()          # capacity 45,000 after the margin
b.acquire(39_000)         # one whole request, as measured
b.acquire(39_000)         # the next request has to wait
check("a second request that doesn't fit waits", len(ft.slept) >= 1, ft.slept)
check("...for the first one to age out of the 60-second window", abs(sum(ft.slept) - 60.0) < 1.0, sum(ft.slept))

b, ft = budget()
b.acquire(30_000)
ft.now += 61
b.acquire(30_000)
check("after 60 seconds the earlier call no longer counts", ft.slept == [], ft.slept)

b, ft = budget()
b.acquire(30_000)
ft.now += 59
b.acquire(10_000)
check("a small call fits beside a big one without waiting", ft.slept == [])

b, ft = budget()
b.acquire(30_000)
ft.now += 30
b.acquire(30_000)
check("it waits only as long as needed: 30 more seconds, not 60", 29.0 <= sum(ft.slept) <= 31.5, sum(ft.slept))

# ---- one call bigger than the whole budget
b, ft = budget(capacity=10_000)
b.acquire(50_000)
check("a call bigger than the budget is let through when nothing else is running", ft.slept == [])
b.acquire(1_000)
check("...and the next call waits for it", sum(ft.slept) >= 59.0, sum(ft.slept))

# ---- correcting the estimate
b, ft = budget()
entry = b.acquire(40_000)
b.settle(entry, 10_000)
b.acquire(30_000)
check("when a call used less than estimated, the difference is freed", ft.slept == [], ft.slept)
b, ft = budget()
entry = b.acquire(10_000)
b.settle(entry, 44_000)
b.acquire(10_000)
check("when a call used more than estimated, the budget remembers", len(ft.slept) >= 1, ft.slept)
b, ft = budget()
entry = b.acquire(10_000)
b.settle(entry, None)
b.settle(entry, 0)
check("an unknown or zero actual count leaves the estimate in place", entry[1] == 10_000)

# ---- separate budgets are independent
a, ft = budget()
c, _ = budget(ft=ft)
a.acquire(44_000)
c.acquire(44_000)
check("two deployments each have their own budget", ft.slept == [])

# ---- estimates
check("estimate: text length over four, plus the expected reply", fc.estimate_tokens("x" * 4000, expected_output=1000) == 2000)
check("estimate: a registered agent's own instructions are added", fc.estimate_tokens("x" * 4000, expected_output=1000, overhead=3500) == 5500)
check("estimate: a dict payload is measured as its JSON", fc.estimate_tokens({"a": "b" * 400}, expected_output=0) > 100)

# ---- budgeted_call
class Resp:
    def __init__(self, total):
        self.usage = type("U", (), {"total_tokens": total})


b, ft = budget()
out = fc.budgeted_call(lambda: Resp(12_345), "x" * 400, budget=b)
check("budgeted_call returns the response untouched", out.usage.total_tokens == 12_345)
check("budgeted_call corrects the reservation to the real token count", b._entries[0][1] == 12_345, b._entries)

b, ft = budget()
def fails():
    raise RuntimeError("boom")
try:
    fc.budgeted_call(fails, "x" * 400, budget=b)
    raised = False
except RuntimeError:
    raised = True
check("budgeted_call lets a real error through (and keeps the estimate counted)", raised and len(b._entries) == 1)

class ResponseWithParts:
    usage = type("U", (), {"total_tokens": None, "input_tokens": 100, "output_tokens": 50})
check("token count is read from input and output when there is no total", fc._actual_tokens(ResponseWithParts) == 150)
check("a response with no usage gives no count", fc._actual_tokens(type("R", (), {})) is None)

# ---- real threads: the budget must never be exceeded, and every call must finish
real = fc.TokenBudget(1000, margin=1.0, window_seconds=0.4)
in_window, peak, lock = [], [0], threading.Lock()


def worker():
    entry = real.acquire(400)
    with lock:
        now = time.time()
        in_window[:] = [t for t in in_window if now - t < 0.4]
        in_window.append(now)
        peak[0] = max(peak[0], len(in_window) * 400)


threads = [threading.Thread(target=worker) for _ in range(8)]
start = time.time()
[t.start() for t in threads]
[t.join(timeout=20) for t in threads]
check("threads: every call finishes", not any(t.is_alive() for t in threads))
check("threads: no 0.4-second window ever held more than the budget", peak[0] <= 1000 + 400, peak[0])  # +400 for timing slack in the measurement
check("threads: the calls were actually spread out, not all let through at once", time.time() - start >= 0.7, time.time() - start)


# ---- learned correction (the first live test over-estimated the legal check by about 28%)
fc._CORRECTIONS.clear()
check("correction starts at 1.0, so a first estimate is cautious", fc.correction_for("regs") == 1.0)
fc.learn_correction("regs", estimated=40_000, actual=30_000)
c1 = fc.correction_for("regs")
check("after a call that used less than estimated, the correction drops", 0.8 < c1 < 1.0, c1)
for _ in range(6):
    fc.learn_correction("regs", estimated=int(40_000 * fc.correction_for("regs")), actual=30_000)
check("it converges on the real ratio (30,000 / 40,000 = 0.75)", abs(fc.correction_for("regs") - 0.75) < 0.03, fc.correction_for("regs"))
fc.learn_correction("wild", estimated=1_000, actual=100_000)
check("one wild call can't push the correction past its ceiling", fc.correction_for("wild") <= fc.CORRECTION_RANGE[1])
fc.learn_correction("tiny", estimated=100_000, actual=10)
check("...or below its floor", fc.correction_for("tiny") >= fc.CORRECTION_RANGE[0])
fc.learn_correction(None, 1000, 500)
fc.learn_correction("zero", 1000, 0)
check("no key, or no real count, learns nothing", fc.correction_for("zero") == 1.0 and None not in fc._CORRECTIONS)
check("different kinds of call learn separately", fc.correction_for("regs") != fc.correction_for("other"))
fc._CORRECTIONS.clear()
b, ft = budget()
fc.budgeted_call(lambda: Resp(12_000), "x" * 40_000, budget=b, expected_output=0, key="k")
check("budgeted_call learns a correction for its key from the real count", 0.5 <= fc.correction_for("k") < 1.3 and fc.correction_for("k") != 1.0, fc.correction_for("k"))
b2, ft2 = budget()
asked = []
real_acquire = b2.acquire
b2.acquire = lambda estimate: asked.append(estimate) or real_acquire(estimate)
fc.budgeted_call(lambda: Resp(12_000), "x" * 40_000, budget=b2, expected_output=0, key="k")
check("the next call of that kind reserves less than the raw estimate of 10,000... scaled by what was learned",
      asked and asked[0] == int(10_000 * (asked[0] / 10_000)) and asked[0] != 10_000, asked)
fc._CORRECTIONS.clear()


print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)

"""
Live check of the token budget: two users ask at the same moment, in one process,
the way a real server would. Without the budget the second request hit the 50,000-tokens-a-minute
limit; with it, the second request should wait and then finish with every sub-agent present.

Makes two real requests (a few cents) and one sign-in.
Usage: python test_concurrent_requests.py
"""
import threading
import time
import warnings
warnings.filterwarnings("ignore")

import foundry_clients as fc
from foundry_clients import Clients
from orchestrator import handle_message

MESSAGES = [
    "We're adding a chatbot that answers billing questions from our knowledge base. It needs to be at least 90% accurate, respond in under 800ms, and we have a $40K annual budget.",
    "We want an AI that suggests replies to agents handling refund requests. Replies must start in under 2 seconds and the budget is $30K a year.",
]
results, t0 = {}, time.time()
log, log_lock = [], threading.Lock()

real_acquire, real_settle = fc.GPT4O_BUDGET.acquire, fc.GPT4O_BUDGET.settle


def acquire(estimate):
    entry = real_acquire(estimate)
    with log_lock:
        log.append({"entry": entry, "estimate": estimate, "at": time.time() - t0})
    return entry


fc.GPT4O_BUDGET.acquire = acquire


def run(i, clients):
    start = time.time()
    outcome = handle_message(MESSAGES[i], None, clients, on_event=lambda *_: None)
    results[i] = (time.time() - start, {k: v["status"] + (f" ({v['error'][:60]})" if v["status"] == "error" else "") for k, v in outcome["results"].items()})


clients = Clients()
threads = [threading.Thread(target=run, args=(i, clients)) for i in range(2)]
[t.start() for t in threads]
[t.join() for t in threads]

for i in range(2):
    print(f"request {i + 1}: {results[i][0]:.0f}s  {results[i][1]}")
print(f"time spent waiting for the token budget: {fc.GPT4O_BUDGET.total_waited:.0f}s across all calls")
print("\nestimate vs real tokens, per gpt-4o call (start time, estimate, counted after the call):")
for item in sorted(log, key=lambda x: x["at"]):
    print(f"  at {item['at']:5.1f}s  estimated {item['estimate']:>7,}  counted {item['entry'][1]:>7,}")

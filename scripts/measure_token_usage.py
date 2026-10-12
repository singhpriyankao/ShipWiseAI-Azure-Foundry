"""
Measures how many tokens each model call in one Orchestrator request uses, and when each call
starts and ends. Written to answer the rate-limit question: the gpt-4o
deployment allows a fixed number of tokens a minute, and the sub-agents run at the same time,
so the sizes and the overlap decide whether staggering them would help or whether the quota
itself is too small.

Usage: python measure_token_usage.py "message"
Makes one real request (a few cents) and one browser sign-in. Prints one line per model call.
"""
import sys
import threading
import time
import warnings
warnings.filterwarnings("ignore")

from foundry_clients import Clients
from orchestrator import handle_message

calls, lock, t0 = [], threading.Lock(), time.time()


def record(kind, name, start, usage):
    with lock:
        calls.append({
            "kind": kind, "name": name, "start": start - t0, "end": time.time() - t0,
            "input": getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None),
            "output": getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None),
        })


def main():
    message = sys.argv[1] if len(sys.argv) > 1 else (
        "We're adding a chatbot that answers billing questions from our knowledge base. "
        "It needs to be at least 90% accurate, respond in under 800ms, and we have a $40K annual budget.")
    clients = Clients()

    agent_responses = clients.agent_client.responses
    original_agent_call = agent_responses.create

    def wrapped_agent(*args, **kwargs):
        start = time.time()
        response = original_agent_call(*args, **kwargs)
        name = (kwargs.get("extra_body") or {}).get("agent_reference", {}).get("name", "agent")
        record("agent", name, start, getattr(response, "usage", None))
        return response
    agent_responses.create = wrapped_agent

    completions = clients.model_client.chat.completions
    original_chat_call = completions.create

    def wrapped_chat(*args, **kwargs):
        start = time.time()
        response = original_chat_call(*args, **kwargs)
        record("chat", kwargs.get("model", "model"), start, getattr(response, "usage", None))
        return response
    completions.create = wrapped_chat

    outcome = handle_message(message, None, clients, on_event=lambda *_: None)
    print("\ncall                                  start   end    input  output")
    for c in sorted(calls, key=lambda c: c["start"]):
        print(f"{c['kind']:5} {c['name'][:30]:30} {c['start']:6.1f} {c['end']:6.1f} {str(c['input']):>8} {str(c['output']):>7}")
    gpt4o = [c for c in calls if c["name"] in ("regulations", "competitive-analysis", "model-stack-analysis", "gpt-4o")]
    print(f"\ntotal tokens, calls that use the gpt-4o deployment: input {sum(c['input'] or 0 for c in gpt4o)}, output {sum(c['output'] or 0 for c in gpt4o)}")
    print("sub-agent results:", {k: v['status'] for k, v in outcome['results'].items()})


if __name__ == "__main__":
    main()

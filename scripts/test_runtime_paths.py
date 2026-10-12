"""
Offline tests for runtime_paths.py and the credential choice in foundry_clients.py. No network, no sign-in.

Run: python test_runtime_paths.py
"""
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import runtime_paths
import foundry_clients

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


def with_env(**values):
    """Set or clear environment variables; returns a function that restores them."""
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    def restore():
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return restore


# ---- paths
restore = with_env(VERCEL=None)
check("locally, a file lives in the scripts folder", runtime_paths.writable_path("price_cache.json") == os.path.join(runtime_paths.HERE, "price_cache.json"))
check("locally, the app is not hosted", runtime_paths.is_hosted() is False)
restore()

restore = with_env(VERCEL="1")
check("on Vercel, the app is hosted", runtime_paths.is_hosted() is True)
path = runtime_paths.writable_path("price_cache.json")
check("on Vercel, a file is in the temporary folder, not the read-only bundle", os.path.dirname(path) == tempfile.gettempdir(), path)
check("on Vercel, a bundled file is copied across the first time", os.path.exists(path) and os.path.getsize(path) == os.path.getsize(os.path.join(runtime_paths.HERE, "price_cache.json")), path)
with open(path, "w") as f:
    f.write("changed by this instance")
check("a second call keeps this instance's own copy and doesn't overwrite it", open(runtime_paths.writable_path("price_cache.json")).read() == "changed by this instance")
os.remove(path)
name = "no_such_bundled_file_xyz.txt"
check("a file with no bundled copy just gets a temporary path", runtime_paths.writable_path(name) == os.path.join(tempfile.gettempdir(), name) and not os.path.exists(runtime_paths.writable_path(name)))
restore()

# ---- credential choice
clean = dict(AZURE_TENANT_ID=None, AZURE_CLIENT_ID=None, AZURE_CLIENT_SECRET=None)
restore = with_env(VERCEL=None, **clean)
check("locally with no app identity, the browser sign-in is used", type(foundry_clients.build_credential()).__name__ == "InteractiveBrowserCredential")
restore()
restore = with_env(VERCEL=None, AZURE_TENANT_ID="t", AZURE_CLIENT_ID="c", AZURE_CLIENT_SECRET="s")
check("with all three app-identity settings, the app identity is used", type(foundry_clients.build_credential()).__name__ == "ClientSecretCredential")
restore()
restore = with_env(VERCEL="1", AZURE_TENANT_ID="t", AZURE_CLIENT_ID="c", AZURE_CLIENT_SECRET=None)
try:
    foundry_clients.build_credential()
    raised = False
except RuntimeError as e:
    raised = "must be set when the app runs hosted" in str(e)
check("hosted with an app-identity setting missing stops with a clear message instead of waiting for a browser", raised)
restore()
restore = with_env(VERCEL=None, AZURE_TENANT_ID="t", AZURE_CLIENT_ID=None, AZURE_CLIENT_SECRET="s")
check("locally with only part of the app identity set, it falls back to the browser sign-in", type(foundry_clients.build_credential()).__name__ == "InteractiveBrowserCredential")
restore()

# ---- cache lifetime
import vendor_prices, model_benchmarks
check("the cache lifetime defaults to 24 hours", vendor_prices.CACHE_TTL_SECONDS == 86400 and model_benchmarks.CACHE_TTL_SECONDS == 86400)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)

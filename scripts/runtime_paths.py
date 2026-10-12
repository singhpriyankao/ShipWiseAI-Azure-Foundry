"""
Where this code may write files. On a laptop that is the scripts folder, as it always was. On Vercel the
deployed code is read-only and only the temporary folder (/tmp, up to 500 MB) is writable, and it is
emptied whenever the platform starts a fresh instance.

`writable_path(name)` returns the scripts-folder path locally. On Vercel (detected by the VERCEL
environment variable the platform sets) it returns a path in the temporary folder and, the first time,
copies the file of the same name from the deployed bundle if there is one. That lets the price and
benchmark caches ship with the deployment as a starting copy, so a fresh instance doesn't have to
download and re-read every vendor page on its first request.
"""
import os
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def is_hosted():
    return bool(os.environ.get("VERCEL"))


def writable_path(filename):
    bundled = os.path.join(HERE, filename)
    if not is_hosted():
        return bundled
    scratch = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(scratch) and os.path.exists(bundled):
        try:
            shutil.copyfile(bundled, scratch)
        except OSError:
            pass
    return scratch

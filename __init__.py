from .ComfyUI_Ollama import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import sys
import os
import subprocess
import threading
import locale


def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors="replace")

    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else:
            print(msg, end="", file=sys.stderr)


def process_wrap(cmd_str, cwd=None):
    process = subprocess.Popen(
        cmd_str,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_thread = threading.Thread(target=handle_stream, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handle_stream, args=(process.stderr, False))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()


pip_install = []

if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
    pip_install = [sys.executable, "-s", "-m", "pip", "install"]
    mim_install = [sys.executable, "-s", "-m", "mim", "install"]
else:
    pip_install = [sys.executable, "-m", "pip", "install"]
    mim_install = [sys.executable, "-m", "mim", "install"]

process_wrap(pip_install + ["-r", "requirements.txt"], os.path.dirname(__file__))

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

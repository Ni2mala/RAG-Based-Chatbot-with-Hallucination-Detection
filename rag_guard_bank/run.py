"""Start the API and the chat UI with one command: `python run.py`.

Ollama must already be running with the models named in `.env`.
Press Ctrl+C to stop both.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "8501"))


def wait_for_api(proc: subprocess.Popen, timeout_s: int = 120) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            requests.get(f"http://127.0.0.1:{API_PORT}/docs", timeout=2)
            return True
        except requests.RequestException:
            time.sleep(1)
    return False


def port_in_use(port: int) -> bool:
    # Try to bind: this also catches a stale server that holds the port without listening.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def main() -> int:
    for port in (API_PORT, UI_PORT):
        if port_in_use(port):
            print(f"Port {port} is already in use. Close the other API or UI window "
                  "(or the program using it) and try again.")
            return 1
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "rag_guard.main:app",
         "--app-dir", "src", "--port", str(API_PORT)],
        cwd=ROOT,
    )
    ui = None
    try:
        print(f"Starting API on port {API_PORT} ...")
        if not wait_for_api(api):
            print("The API did not start; see the messages above.")
            return 1
        print(f"Starting chat UI at http://localhost:{UI_PORT} ...")
        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py",
             "--server.port", str(UI_PORT), "--server.address", "localhost"],
            cwd=ROOT,
            env={**os.environ, "RAG_API_URL": f"http://127.0.0.1:{API_PORT}"},
        )
        while api.poll() is None and ui.poll() is None:
            time.sleep(1)
        return 1
    except KeyboardInterrupt:
        print("\nStopping ...")
        return 0
    finally:
        for proc in (ui, api):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    sys.exit(main())

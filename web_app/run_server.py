"""
run_server.py — Launcher script for the Aegis MetroHealth FastAPI backend.
Run from the main_implementation/ root:
    .venv\\Scripts\\python.exe web_app\\run_server.py
"""
import sys
import os

# Add both project root and backend dir so 'app' and 'src' resolve correctly
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_ROOT, "web_app", "backend")

# Inject paths into PYTHONPATH env so child reload processes inherit them
os.environ["PYTHONPATH"] = os.pathsep.join([_ROOT, _BACKEND]) + os.pathsep + os.environ.get("PYTHONPATH", "")

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[_BACKEND],
    )

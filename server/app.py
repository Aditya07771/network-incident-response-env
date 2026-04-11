"""
Thin OpenEnv server entrypoint wrapper.

This preserves the existing root-level FastAPI app while also exposing
the `server.app:main` entrypoint expected by `openenv validate`.
"""

try:
    from app import app as app
except ImportError:
    import sys
    import os
    # Add the parent directory to sys.path so 'app' can be found
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app import app as app


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
"""
Thin OpenEnv server entrypoint wrapper.

This preserves the existing root-level FastAPI app while also exposing
the `server.app:main` entrypoint expected by `openenv validate`.
"""

from app import app as app


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

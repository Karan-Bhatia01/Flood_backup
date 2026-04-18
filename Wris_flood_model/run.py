"""
Entry point — run this to start the Flood Prediction API server.
Usage:
    python run.py
    
Or directly with uvicorn:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from dotenv import load_dotenv

load_dotenv()   # reads .env file

import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    print("=" * 60)
    print("  🌊 Flood Risk Prediction API")
    print(f"  Server  : http://{host}:{port}")
    print(f"  Docs    : http://localhost:{port}/docs")
    print(f"  Health  : http://localhost:{port}/health")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,      # auto-reload on code changes (dev mode)
        log_level="info",
    )

#!/usr/bin/env python3
"""Entry point for running the Coral Model Convert service"""

import uvicorn
from coral_model_convert.main import app

if __name__ == "__main__":
    uvicorn.run(
        "coral_model_convert.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
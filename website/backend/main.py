"""
OptiSkill Demo 後端入口。

    cd website/backend
    uvicorn main:app --reload --port 8000
"""

import logging

from app.server import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()

# run.py

from app.routes import app
import hypercorn.asyncio
from hypercorn.config import Config
import asyncio

if __name__ == '__main__':
    config = Config()
    config.bind = ["0.0.0.0:8000"]  # Listen on all interfaces
    config.workers = 2  # Number of worker processes
    config.loglevel = "info"  # Logging level
    config.use_reloader = False  # Disable auto-reload in production
    asyncio.run(hypercorn.asyncio.serve(app, config))

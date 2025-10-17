import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import chunks, ingest, retrieve, sync
from graph_service.zep_graphiti import initialize_graphiti


def setup_logging(log_file: str | None):
    """Configure logging to file only (stderr captured separately by launcher)

    Application logs go to file via FileHandler.
    Python errors/tracebacks go to stderr, which the Cymbiont launcher
    redirects to the same log file for consolidated chronological output.
    """
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),
            ]
        )
    else:
        # Default logging to stdout only
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_file)
    await initialize_graphiti(settings)
    yield
    # Shutdown
    # No need to close Graphiti here, as it's handled per-request


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)
app.include_router(sync.router)
app.include_router(chunks.router, prefix='/chunks', tags=['chunks'])


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)

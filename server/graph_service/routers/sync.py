"""Document synchronization endpoints."""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, status
from graphiti_core.document_sync import DocumentSyncManager, DocumentWatcher

from graph_service.dto import (
    StartSyncRequest,
    SyncActionResponse,
    SyncStatsResponse,
    SyncStatusResponse,
)
from graph_service.zep_graphiti import ZepGraphitiDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/sync', tags=['sync'])

# Module-level state for document sync
# These persist across requests (unlike per-request Graphiti clients)
sync_manager: DocumentSyncManager | None = None
document_watcher: DocumentWatcher | None = None


@router.post('/start', status_code=status.HTTP_200_OK)
async def start_sync(
    request: StartSyncRequest,
    graphiti: ZepGraphitiDep,
) -> SyncActionResponse:
    """Start document watcher for corpus directory.

    Idempotent: safe to call if already running.
    """
    global sync_manager, document_watcher

    # If already running, return success
    if document_watcher is not None and document_watcher._running:
        return SyncActionResponse(
            status='already_running',
            message=f'Document watcher already active for {sync_manager.corpus_path}',
            corpus_path=str(sync_manager.corpus_path) if sync_manager else None,
            sync_interval=document_watcher.sync_interval / 3600
            if document_watcher
            else None,  # Convert seconds to hours
        )

    # Create sync manager
    corpus_path = Path(request.corpus_path)
    if not corpus_path.exists():
        return SyncActionResponse(
            status='error',
            message=f'Corpus path does not exist: {request.corpus_path}',
        )

    sync_manager = DocumentSyncManager(
        corpus_path=corpus_path,
        graphiti=graphiti,
        group_id=request.group_id,
    )

    # Create and start document watcher
    document_watcher = DocumentWatcher(
        sync_manager=sync_manager,
        sync_interval_hours=request.sync_interval_hours,
    )

    # Get current event loop
    loop = asyncio.get_event_loop()
    document_watcher.start(loop)

    logger.info(
        f'Document watcher started for {corpus_path} '
        f'(sync interval: {request.sync_interval_hours}h, group_id: {request.group_id})'
    )

    return SyncActionResponse(
        status='started',
        message=f'Document watcher started for {corpus_path}',
        corpus_path=str(corpus_path),
        sync_interval=request.sync_interval_hours,
    )


@router.post('/stop', status_code=status.HTTP_200_OK)
async def stop_sync() -> SyncActionResponse:
    """Stop document watcher.

    Idempotent: safe to call if not running.
    """
    global sync_manager, document_watcher

    if document_watcher is None or not document_watcher._running:
        return SyncActionResponse(
            status='not_running',
            message='Document watcher is not active',
        )

    # Stop watcher
    await document_watcher.stop()
    corpus_path = str(sync_manager.corpus_path) if sync_manager else None

    # Clear state
    document_watcher = None
    sync_manager = None

    logger.info(f'Document watcher stopped for {corpus_path}')

    return SyncActionResponse(
        status='stopped',
        message=f'Document watcher stopped for {corpus_path}',
    )


@router.post('/trigger', status_code=status.HTTP_200_OK)
async def trigger_sync(async_mode: bool = True) -> SyncActionResponse | SyncStatsResponse:
    """Manually trigger document synchronization.

    Args:
        async_mode: If True (default), returns immediately and runs sync in background.
                   If False, waits for sync to complete and returns full stats.
    """
    global sync_manager

    if sync_manager is None:
        return SyncActionResponse(
            status='error',
            message='Document sync not initialized (call /sync/start first)',
        )

    if async_mode:
        # Run sync in background
        async def run_sync():
            try:
                result = await sync_manager.sync_all()
                logger.info(
                    f'Manual sync completed: {result["synced"]} synced, '
                    f'{result["skipped"]} skipped, {len(result["errors"])} errors'
                )
            except Exception as e:
                logger.error(f'Error during manual sync: {e}')

        asyncio.create_task(run_sync())

        return SyncActionResponse(
            status='triggered',
            message='Document sync started in background',
        )
    else:
        # Run sync synchronously and return stats
        result = await sync_manager.sync_all()
        logger.info(
            f'Manual sync completed: {result["synced"]} synced, '
            f'{result["skipped"]} skipped, {len(result["errors"])} errors'
        )
        return SyncStatsResponse(
            synced=result['synced'],
            skipped=result['skipped'],
            errors=result['errors'],
        )


@router.get('/status', status_code=status.HTTP_200_OK)
async def get_sync_status() -> SyncStatusResponse:
    """Get document watcher status."""
    global sync_manager, document_watcher

    if document_watcher is None or not document_watcher._running:
        return SyncStatusResponse(
            running=False,
            corpus_path=None,
            sync_interval=None,
        )

    return SyncStatusResponse(
        running=True,
        corpus_path=str(sync_manager.corpus_path) if sync_manager else None,
        sync_interval=document_watcher.sync_interval / 3600 if document_watcher else None,
    )

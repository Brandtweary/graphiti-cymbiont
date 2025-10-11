"""DTOs for document sync endpoints."""

from pydantic import BaseModel, Field


class StartSyncRequest(BaseModel):
    """Request to start document watcher."""

    corpus_path: str = Field(..., description='Absolute path to corpus directory')
    sync_interval_hours: float = Field(
        default=1.0, description='Hours between batch syncs (default: 1.0)'
    )
    group_id: str = Field(default='default', description='Group ID for synced episodes')


class SyncStatusResponse(BaseModel):
    """Document watcher status."""

    running: bool = Field(..., description='Whether document watcher is active')
    corpus_path: str | None = Field(None, description='Corpus directory path if running')
    sync_interval: float | None = Field(None, description='Sync interval in hours if running')


class SyncStatsResponse(BaseModel):
    """Document sync statistics."""

    synced: int = Field(..., description='Number of documents synced')
    skipped: int = Field(..., description='Number of documents skipped (unchanged)')
    errors: list[dict] = Field(..., description='List of error details')
    total: int = Field(..., description='Total documents processed')


class SyncActionResponse(BaseModel):
    """Response for sync start/stop actions."""

    status: str = Field(..., description='Action result (started, stopped, already_running, etc.)')
    message: str = Field(..., description='Human-readable status message')
    corpus_path: str | None = Field(None, description='Corpus path if applicable')
    sync_interval: float | None = Field(None, description='Sync interval if applicable')

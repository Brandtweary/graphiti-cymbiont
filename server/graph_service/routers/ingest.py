import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial

from fastapi import APIRouter, FastAPI, Query, status
from graphiti_core.nodes import DocumentNode, ChunkNode, EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore
from transformers import AutoTokenizer

from graph_service.dto import (
    AddEntityNodeRequest,
    AddEpisodeRequest,
    AddMessagesRequest,
    Message,
    Result,
)
from graph_service.zep_graphiti import ZepGraphitiDep

# Tokenizer for chunk token counting (same as chunker.py)
TOKENIZER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None

    async def worker(self):
        while True:
            try:
                print(f'Got a job: (size of remaining queue: {self.queue.qsize()})')
                job = await self.queue.get()
                await job()
            except asyncio.CancelledError:
                break

    async def start(self):
        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            await self.task
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await async_worker.start()
    yield
    await async_worker.stop()


router = APIRouter(lifespan=lifespan)


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    async def add_messages_task(m: Message):
        await graphiti.add_episode(
            uuid=m.uuid,
            group_id=request.group_id,
            name=m.name,
            episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
        )

    for m in request.messages:
        await async_worker.queue.put(partial(add_messages_task, m))

    return Result(message='Messages added to processing queue', success=True)


@router.post('/episodes', status_code=status.HTTP_202_ACCEPTED)
async def add_episode(
    request: AddEpisodeRequest,
    graphiti: ZepGraphitiDep,
):
    # Map source string to EpisodeType enum
    source_type = EpisodeType.text
    if request.source.lower() == 'message':
        source_type = EpisodeType.message
    elif request.source.lower() == 'json':
        source_type = EpisodeType.json

    async def add_episode_task():
        # Add episode and get result with UUID
        result = await graphiti.add_episode(
            uuid=request.uuid,
            group_id=request.group_id,
            name=request.name,
            episode_body=request.episode_body,
            reference_time=datetime.now(timezone.utc),
            source=source_type,
            source_description=request.source_description,
        )

        # Create ChunkNode for manual episode
        episode_uuid = result.episode.uuid
        episode_body = request.episode_body

        # Count tokens using tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        tokens = tokenizer.encode(episode_body, add_special_tokens=False)
        token_count = len(tokens)

        # Create single chunk for manual episode
        chunk = ChunkNode(
            name=request.name,
            content=episode_body,
            document_uri=f'episode:{episode_uuid}',
            chunk_index=0,
            total_chunks=1,
            token_count=token_count,
            group_id=request.group_id,
        )

        # Save chunk to Neo4j
        await chunk.save(graphiti.driver)

        # Create [:GENERATED_FROM] relationship: (chunk)-[:GENERATED_FROM]->(episode)
        await graphiti.driver.execute_query(
            """
            MATCH (c:Chunk {uuid: $chunk_uuid})
            MATCH (e:Episodic {uuid: $episode_uuid})
            MERGE (c)-[:GENERATED_FROM]->(e)
            """,
            chunk_uuid=chunk.uuid,
            episode_uuid=episode_uuid,
        )

    await async_worker.queue.put(add_episode_task)
    return Result(message='Episode added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    """Delete an episode and its associated chunk (for manual episodes only).

    For manual episodes (created via /episodes endpoint), this also deletes the
    associated ChunkNode with document_uri='episode:{uuid}'.

    For document sync episodes, chunks are managed separately through the document
    sync pipeline and are not deleted when the episode is deleted.
    """
    # Delete associated chunk if this is a manual episode
    # Manual episodes have chunks with document_uri='episode:{uuid}'
    await graphiti.driver.execute_query(
        """
        MATCH (c:Chunk {document_uri: $document_uri})
        DETACH DELETE c
        """,
        document_uri=f'episode:{uuid}',
    )

    # Delete the episode
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.delete('/document/{uri:path}', status_code=status.HTTP_200_OK)
async def delete_document(
    uri: str,
    graphiti: ZepGraphitiDep,
    group_id: str = Query(default='default', description='Group ID for the document'),
    delete_episodes: bool = Query(default=False, description='Also delete associated episodes'),
):
    """Delete a document and its chunks from the knowledge graph.

    Args:
        uri: Document URI (relative path from corpus root)
        group_id: Group ID (defaults to 'default')
        delete_episodes: If True, also delete all episodes associated with this document

    Returns:
        Result indicating success with counts of deleted items
    """

    # Get document node
    doc_node = await DocumentNode.get_by_uri(graphiti.driver, uri, group_id)
    if not doc_node:
        return Result(message=f'Document not found: {uri}', success=False)

    # Delete all chunks (query with count since delete_by_document_uri doesn't return count)
    records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (c:Chunk {document_uri: $document_uri, group_id: $group_id})
        WITH count(c) as chunk_count
        MATCH (c:Chunk {document_uri: $document_uri, group_id: $group_id})
        DETACH DELETE c
        RETURN chunk_count
        """,
        document_uri=uri,
        group_id=group_id,
    )
    deleted_chunks = records[0].get('chunk_count', 0) if records else 0

    # Optionally delete episodes
    deleted_episodes = 0
    if delete_episodes:
        # Query for all episodes with this document URI in metadata
        query = """
        MATCH (e:Episodic {group_id: $group_id})
        WHERE e.metadata CONTAINS $uri
        DETACH DELETE e
        RETURN count(e) as deleted
        """
        records, _, _ = await graphiti.driver.execute_query(query, group_id=group_id, uri=uri)
        for record in records:
            deleted_episodes = record.get('deleted', 0)

    # Delete document node (manual query since Node.delete() doesn't handle Document label)
    await graphiti.driver.execute_query(
        """
        MATCH (d:Document {uuid: $uuid, group_id: $group_id})
        DETACH DELETE d
        """,
        uuid=doc_node.uuid,
        group_id=group_id,
    )

    return Result(
        message=f'Document deleted: {uri} ({deleted_chunks} chunks, {deleted_episodes} episodes)',
        success=True,
    )


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)

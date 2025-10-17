"""Core document synchronization logic.

This module manages the synchronization of markdown documents from a corpus directory
into the Graphiti knowledge graph. It implements intelligent change detection with
document chunking for efficient knowledge extraction.

Key Concepts:
    Three-Tier Architecture:
        - DocumentNode: Document index with change detection (content hash, timestamps)
        - EpisodicNode: Append-only history (chunk episodes, diff episodes)
        - ChunkNode: Retrieval-optimized text fragments (BM25 search)

    Change Detection:
        - DocumentNode-based (replaces episode-based lookup)
        - SHA256 content hashing to detect actual changes
        - Skip sync if content unchanged
        - Lazy migration for unchunked documents (last_chunk_at == None)

    Initial Sync (New Documents):
        - Chunk document using Docling HybridChunker (2000 tokens, structure-aware)
        - Create episode per chunk (goes through LLM extraction)
        - Create DocumentNode with content/hash
        - Create ChunkNodes for retrieval

    Update Sync (Changed Documents):
        - Generate unified diff (old vs new content from DocumentNode)
        - Create diff episode (LLM summarized, <2000 tokens)
        - Update DocumentNode (content, hash, timestamps)
        - Regenerate chunks (delete old, create new ChunkNodes)

    Rename Detection:
        - Query DocumentNode by content_hash with different URI
        - Update DocumentNode.uri (single property update)
        - Check if content also changed after rename

    File Operations:
        - Rename: Update DocumentNode URI
        - Delete: No-op (append-only graph preserves history)

Episode Metadata Schema:
    {
        "document_uri": str,      # Relative to corpus root
        "content_hash": str,      # SHA256 with prefix
        "sync_type": str,         # "chunk" | "diff"
        "chunk_index": int,       # (chunk episodes only)
        "total_chunks": int,      # (chunk episodes only)
        "sync_timestamp": str     # ISO 8601 UTC timestamp
    }
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.helpers import semaphore_gather
from graphiti_core.nodes import ChunkNode, DocumentNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now

from .chunker import chunk_document_for_episodes, chunk_document_for_retrieval
from .diff_generator import compute_content_hash, generate_unified_diff
from .diff_summarizer import summarize_diff

logger = logging.getLogger(__name__)


class DocumentSyncManager:
    """Manages synchronization of documents from corpus directory to knowledge graph."""

    def __init__(
        self,
        corpus_path: Path,
        graphiti: Graphiti,
        group_id: str,
    ):
        """Initialize document sync manager.

        Args:
            corpus_path: Root directory containing documents to sync
            graphiti: Graphiti client instance
            group_id: Graph partition identifier
        """
        self.corpus_path = Path(corpus_path)
        self.graphiti = graphiti
        self.group_id = group_id

    def _get_relative_uri(self, file_path: Path) -> str:
        """Convert absolute file path to relative URI.

        Args:
            file_path: Absolute path to file

        Returns:
            Relative path from corpus root (e.g., 'tasks/active.md')
        """
        return str(file_path.relative_to(self.corpus_path))

    async def get_latest_episode_for_document(self, uri: str) -> EpisodicNode | None:
        """Query graph for most recent episode with matching document URI.

        Args:
            uri: Document URI to search for

        Returns:
            Latest EpisodicNode for this document, or None if not found
        """
        # Build Cypher query to find episodes with this URI in metadata
        # Use string matching on JSON since metadata is stored as JSON string
        # Pattern: "document_uri": "uri_value"
        search_pattern = f'"document_uri": "{uri}"'

        query = """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
          AND e.metadata IS NOT NULL
          AND e.metadata CONTAINS $search_pattern
        RETURN e
        ORDER BY e.created_at DESC
        LIMIT 1
        """

        try:
            # execute_query returns (records, summary, keys)
            records, _, _ = await self.graphiti.driver.execute_query(
                query,
                group_id=self.group_id,
                search_pattern=search_pattern,
            )

            if records and len(records) > 0:
                # Parse the episode node from the result
                record = records[0]
                episode_data = record['e']

                # Convert Neo4j datetime objects to Python datetime
                # Neo4j returns neo4j.time.DateTime which needs conversion
                created_at = episode_data['created_at']
                if hasattr(created_at, 'to_native'):
                    created_at = created_at.to_native()

                valid_at = episode_data['valid_at']
                if hasattr(valid_at, 'to_native'):
                    valid_at = valid_at.to_native()

                # Convert Neo4j node to EpisodicNode
                return EpisodicNode(
                    uuid=episode_data['uuid'],
                    name=episode_data['name'],
                    group_id=episode_data['group_id'],
                    created_at=created_at,
                    source=EpisodeType(episode_data['source']),
                    source_description=episode_data.get('source_description', ''),
                    content=episode_data['content'],
                    valid_at=valid_at,
                    entity_edges=episode_data.get('entity_edges', []),
                    metadata=json.loads(episode_data['metadata'])
                    if episode_data.get('metadata')
                    else {},
                )
        except Exception as e:
            logger.error(f'Error querying for document {uri}: {e}')
            return None

        return None

    async def get_latest_full_sync_for_document(self, uri: str) -> EpisodicNode | None:
        """Get most recent full sync episode for document (authoritative snapshot).

        Args:
            uri: Document URI relative to corpus root

        Returns:
            Latest full sync EpisodicNode for this document, or None if not found
        """
        search_pattern = f'"document_uri": "{uri}"'
        full_sync_pattern = f'"sync_type": "full"'

        query = """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
          AND e.metadata IS NOT NULL
          AND e.metadata CONTAINS $search_pattern
          AND e.metadata CONTAINS $full_sync_pattern
        RETURN e
        ORDER BY e.created_at DESC
        LIMIT 1
        """

        try:
            records, _, _ = await self.graphiti.driver.execute_query(
                query,
                group_id=self.group_id,
                search_pattern=search_pattern,
                full_sync_pattern=full_sync_pattern,
            )

            if records and len(records) > 0:
                record = records[0]
                episode_data = record['e']

                # Convert Neo4j datetime objects
                created_at = episode_data['created_at']
                if hasattr(created_at, 'to_native'):
                    created_at = created_at.to_native()

                valid_at = episode_data['valid_at']
                if hasattr(valid_at, 'to_native'):
                    valid_at = valid_at.to_native()

                return EpisodicNode(
                    uuid=episode_data['uuid'],
                    name=episode_data['name'],
                    group_id=episode_data['group_id'],
                    created_at=created_at,
                    source=EpisodeType(episode_data['source']),
                    source_description=episode_data.get('source_description', ''),
                    content=episode_data['content'],
                    valid_at=valid_at,
                    entity_edges=episode_data.get('entity_edges', []),
                    metadata=json.loads(episode_data['metadata'])
                    if episode_data.get('metadata')
                    else {},
                )
        except Exception as e:
            logger.error(f'Error querying for full sync of document {uri}: {e}')
            return None

        return None

    async def find_episode_by_content_hash(
        self, content_hash: str, exclude_uri: str
    ) -> EpisodicNode | None:
        """Find episode with matching content hash but different URI (indicates rename).

        This enables rename detection even when watchdog events are missed (server downtime,
        bulk operations, etc). If a file with identical content appears at a new location,
        it's likely a rename rather than a new document.

        Args:
            content_hash: SHA256 hash to search for
            exclude_uri: Current URI to exclude from search

        Returns:
            Episode with matching hash from different URI, or None if not found
        """
        # Search for episodes with this content hash
        hash_pattern = f'"content_hash": "{content_hash}"'
        # Exclude episodes with the current URI
        exclude_pattern = f'"document_uri": "{exclude_uri}"'

        query = """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
          AND e.metadata IS NOT NULL
          AND e.metadata CONTAINS $hash_pattern
          AND NOT e.metadata CONTAINS $exclude_pattern
        RETURN e
        ORDER BY e.created_at DESC
        LIMIT 1
        """

        try:
            records, _, _ = await self.graphiti.driver.execute_query(
                query,
                group_id=self.group_id,
                hash_pattern=hash_pattern,
                exclude_pattern=exclude_pattern,
            )

            if records and len(records) > 0:
                record = records[0]
                episode_data = record['e']

                # Convert Neo4j datetime objects
                created_at = episode_data['created_at']
                if hasattr(created_at, 'to_native'):
                    created_at = created_at.to_native()

                valid_at = episode_data['valid_at']
                if hasattr(valid_at, 'to_native'):
                    valid_at = valid_at.to_native()

                return EpisodicNode(
                    uuid=episode_data['uuid'],
                    name=episode_data['name'],
                    group_id=episode_data['group_id'],
                    created_at=created_at,
                    source=EpisodeType(episode_data['source']),
                    source_description=episode_data.get('source_description', ''),
                    content=episode_data['content'],
                    valid_at=valid_at,
                    entity_edges=episode_data.get('entity_edges', []),
                    metadata=json.loads(episode_data['metadata'])
                    if episode_data.get('metadata')
                    else {},
                )
        except Exception as e:
            logger.error(f'Error querying for content hash {content_hash}: {e}')
            return None

        return None

    async def sync_document(self, file_path: Path) -> dict[str, Any]:
        """Sync a single document to the knowledge graph.

        Uses DocumentNode for change detection and chunking for episode creation.
        Creates chunk episodes on initial sync and diff episodes on updates.

        Args:
            file_path: Absolute path to document file

        Returns:
            Sync result with status and metadata
        """
        uri = self._get_relative_uri(file_path)

        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            new_hash = compute_content_hash(content)

            # Query for DocumentNode (replaces episode-based lookup)
            doc_node = await DocumentNode.get_by_uri(self.graphiti.driver, uri, self.group_id)

            # Check if document exists
            if doc_node:
                # Lazy migration: check if document was never chunked
                if doc_node.last_chunk_at is None:
                    logger.info(
                        f'Document {uri} - lazy migration detected (no chunks), generating chunks'
                    )
                    # Force re-chunk even if content unchanged
                    # This handles migration from pre-chunking era
                    chunks = chunk_document_for_retrieval(str(file_path), uri)
                    logger.info(f'Document {uri} - created {len(chunks)} retrieval chunks (lazy migration)')

                    reference_time = utc_now()
                    for chunk in chunks:
                        chunk_node = ChunkNode(
                            name=f'{uri}_chunk_{chunk.chunk_index}',
                            content=chunk.content,
                            chunk_index=chunk.chunk_index,
                            total_chunks=chunk.total_chunks,
                            token_count=chunk.token_count,
                            document_uri=uri,
                            group_id=self.group_id,
                            created_at=reference_time,
                        )
                        await chunk_node.save(self.graphiti.driver)

                    # Update last_chunk_at
                    doc_node.last_chunk_at = reference_time
                    await doc_node.save(self.graphiti.driver)
                    logger.info(f'Document {uri} - lazy migration complete')

                # Document exists - check if content changed
                if doc_node.content_hash == new_hash:
                    logger.info(f'Document {uri} - SKIPPED: content unchanged')
                    return {
                        'status': 'skipped',
                        'uri': uri,
                        'reason': 'unchanged',
                    }

                # Content changed - update flow
                logger.info(f'Document {uri} - content changed, processing update')
                return await self._handle_document_update(
                    file_path, uri, content, new_hash, doc_node
                )
            else:
                # No DocumentNode - check for rename via content hash
                renamed_doc = await DocumentNode.find_by_content_hash(
                    self.graphiti.driver, new_hash, self.group_id
                )

                if renamed_doc and renamed_doc.uri != uri:
                    # Rename detected
                    old_uri = renamed_doc.uri
                    logger.info(
                        f'Document {uri} - detected rename via content hash: {old_uri} → {uri}'
                    )

                    # Update DocumentNode URI
                    renamed_doc.uri = uri
                    await renamed_doc.save(self.graphiti.driver)

                    # Check if content also changed after rename
                    if renamed_doc.content_hash == new_hash:
                        logger.info(f'Document {uri} - SKIPPED: renamed, content unchanged')
                        return {
                            'status': 'skipped',
                            'uri': uri,
                            'reason': 'renamed_unchanged',
                        }

                    # Content changed after rename
                    logger.info(f'Document {uri} - content changed after rename')
                    return await self._handle_document_update(
                        file_path, uri, content, new_hash, renamed_doc
                    )
                else:
                    # New document - initial sync
                    logger.info(f'Document {uri} - first sync, performing initial chunking')
                    return await self._handle_initial_sync(file_path, uri, content, new_hash)

        except Exception as e:
            logger.error(f'Error syncing document {uri}: {e}')
            return {
                'status': 'error',
                'uri': uri,
                'error': str(e),
            }

    async def _handle_initial_sync(
        self, file_path: Path, uri: str, content: str, content_hash: str
    ) -> dict[str, Any]:
        """Handle initial sync of a new document with chunking.

        Chunks the document and creates an episode for each chunk.
        Then creates DocumentNode and ChunkNodes.

        Args:
            file_path: Absolute path to document file
            uri: Document URI
            content: Document content
            content_hash: SHA256 hash of content

        Returns:
            Sync result dictionary
        """
        # Stage 1: Chunk for episode ingestion (large chunks for LLM context)
        episode_chunks = chunk_document_for_episodes(str(file_path), uri)

        # Log episode chunk details
        if episode_chunks:
            chunk_sizes = [c.token_count for c in episode_chunks]
            logger.info(
                f'Document {uri} - created {len(episode_chunks)} episode chunks for LLM ingestion: '
                f'sizes={chunk_sizes[0]}-{chunk_sizes[-1]} tokens, '
                f'avg={sum(chunk_sizes)/len(chunk_sizes):.0f} tokens'
            )
        else:
            logger.info(f'Document {uri} - no episode chunks created (empty document?)')

        # Create episode for each chunk (goes through LLM extraction)
        reference_time = utc_now()
        for chunk in episode_chunks:
            episode_name = f'Document chunk: {uri} [{chunk.chunk_index + 1}/{chunk.total_chunks}]'
            episode_body = f'Document: {uri}\nChunk {chunk.chunk_index + 1} of {chunk.total_chunks}\n\n{chunk.content}'

            metadata = {
                'document_uri': uri,
                'content_hash': content_hash,
                'sync_type': 'chunk',
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks,
                'sync_timestamp': reference_time.isoformat(),
            }

            await self.graphiti.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source_description=f'Document chunk from {uri}',
                reference_time=reference_time,
                source=EpisodeType.text,
                group_id=self.group_id,
                metadata=metadata,
            )

        # Create DocumentNode
        doc_node = DocumentNode(
            name=uri,
            uri=uri,
            content=content,
            content_hash=content_hash,
            last_sync_at=reference_time,
            last_chunk_at=reference_time,
            group_id=self.group_id,
            created_at=reference_time,
        )
        await doc_node.save(self.graphiti.driver)
        logger.info(f'Document {uri} - created DocumentNode')

        # Stage 2: Chunk for retrieval (small chunks for search)
        retrieval_chunks = chunk_document_for_retrieval(str(file_path), uri)

        # Create ChunkNodes
        for chunk in retrieval_chunks:
            chunk_node = ChunkNode(
                name=f'{uri}_chunk_{chunk.chunk_index}',
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                token_count=chunk.token_count,
                document_uri=uri,
                group_id=self.group_id,
                created_at=reference_time,
            )
            await chunk_node.save(self.graphiti.driver)

        logger.info(f'Document {uri} - created {len(retrieval_chunks)} retrieval ChunkNodes')
        logger.info(f'Document {uri} - initial sync complete')

        return {
            'status': 'synced',
            'uri': uri,
            'sync_type': 'initial',
            'content_hash': content_hash,
            'episode_chunk_count': len(episode_chunks),
            'retrieval_chunk_count': len(retrieval_chunks),
        }

    async def _handle_document_update(
        self,
        file_path: Path,
        uri: str,
        content: str,
        content_hash: str,
        doc_node: DocumentNode,
    ) -> dict[str, Any]:
        """Handle update of existing document.

        Generates diff, chunks it (usually 1 chunk), and creates diff episodes.
        Always regenerates ChunkNodes from new content.

        Args:
            file_path: Absolute path to document file
            uri: Document URI
            content: New document content
            content_hash: SHA256 hash of new content
            doc_node: Existing DocumentNode

        Returns:
            Sync result dictionary
        """
        # Generate diff
        old_content = doc_node.content
        diff_content = generate_unified_diff(old_content, content, uri)

        # Chunk the diff (reuse Docling to ensure episodes don't exceed LLM context window)
        # Write diff to temp file for chunking
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
            tmp_file.write(diff_content)
            tmp_path = tmp_file.name

        try:
            diff_chunks = chunk_document_for_episodes(tmp_path, uri)

            # Log diff episode chunk details
            if diff_chunks:
                chunk_sizes = [c.token_count for c in diff_chunks]
                logger.info(
                    f'Document {uri} - chunked diff into {len(diff_chunks)} episode chunks: '
                    f'sizes={chunk_sizes[0]}-{chunk_sizes[-1]} tokens, '
                    f'avg={sum(chunk_sizes)/len(chunk_sizes):.0f} tokens'
                )
            else:
                logger.info(f'Document {uri} - no diff chunks (empty diff?)')
        finally:
            # Clean up temp file
            Path(tmp_path).unlink()

        # Create episode for each diff chunk
        reference_time = utc_now()
        for chunk in diff_chunks:
            episode_name = f'Document diff: {uri}'
            if len(diff_chunks) > 1:
                episode_name += f' [{chunk.chunk_index + 1}/{chunk.total_chunks}]'

            # Summarize this diff chunk
            summary = await summarize_diff(
                self.graphiti.llm_client,
                chunk.content,
                uri,
            )

            metadata = {
                'document_uri': uri,
                'content_hash': content_hash,
                'sync_type': 'diff',
                'chunk_index': chunk.chunk_index if len(diff_chunks) > 1 else None,
                'total_chunks': chunk.total_chunks if len(diff_chunks) > 1 else None,
                'sync_timestamp': reference_time.isoformat(),
            }

            # Create diff episode (append-only)
            await self.graphiti.add_episode(
                name=episode_name,
                episode_body=summary,
                source_description=f'Document diff from {uri}',
                reference_time=reference_time,
                source=EpisodeType.text,
                group_id=self.group_id,
                metadata=metadata,
            )

        # Update DocumentNode
        doc_node.content = content
        doc_node.content_hash = content_hash
        doc_node.last_sync_at = reference_time
        doc_node.last_chunk_at = reference_time
        await doc_node.save(self.graphiti.driver)
        logger.info(f'Document {uri} - updated DocumentNode')

        # Regenerate ChunkNodes from new content
        # Delete old chunks
        await ChunkNode.delete_by_document_uri(self.graphiti.driver, uri, self.group_id)
        logger.info(f'Document {uri} - deleted old chunks')

        # Create new retrieval chunks
        chunks = chunk_document_for_retrieval(str(file_path), uri)
        logger.info(f'Document {uri} - created {len(chunks)} new retrieval chunks')

        for chunk in chunks:
            chunk_node = ChunkNode(
                name=f'{uri}_chunk_{chunk.chunk_index}',
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                token_count=chunk.token_count,
                document_uri=uri,
                group_id=self.group_id,
                created_at=reference_time,
            )
            await chunk_node.save(self.graphiti.driver)

        logger.info(f'Document {uri} - update complete')

        return {
            'status': 'synced',
            'uri': uri,
            'sync_type': 'diff',
            'content_hash': content_hash,
            'retrieval_chunk_count': len(chunks),
            'diff_episode_count': len(diff_chunks),
        }

    async def handle_rename(self, old_uri: str, new_uri: str) -> dict[str, Any]:
        """Update all episode metadata URIs when file is renamed.

        Args:
            old_uri: Previous document URI
            new_uri: New document URI

        Returns:
            Result with count of updated episodes
        """
        # Find all episodes with the old URI
        # We'll update them one by one using Python to avoid APOC dependency
        search_pattern = f'"document_uri": "{old_uri}"'

        query_find = """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
          AND e.metadata IS NOT NULL
          AND e.metadata CONTAINS $search_pattern
        RETURN e.uuid as uuid, e.metadata as metadata
        """

        try:
            # Find episodes to update
            # execute_query returns (records, summary, keys)
            records, _, _ = await self.graphiti.driver.execute_query(
                query_find,
                group_id=self.group_id,
                search_pattern=search_pattern,
            )

            if not records:
                return {
                    'status': 'renamed',
                    'old_uri': old_uri,
                    'new_uri': new_uri,
                    'updated_count': 0,
                }

            # Update each episode's metadata
            updated_count = 0
            for record in records:
                episode_uuid = record['uuid']
                old_metadata_json = record['metadata']

                # Parse metadata, update URI, re-serialize
                metadata = json.loads(old_metadata_json)
                metadata['document_uri'] = new_uri
                new_metadata_json = json.dumps(metadata)

                # Update the episode
                query_update = """
                MATCH (e:Episodic)
                WHERE e.uuid = $uuid
                SET e.metadata = $new_metadata
                """

                await self.graphiti.driver.execute_query(
                    query_update,
                    uuid=episode_uuid,
                    new_metadata=new_metadata_json,
                )
                updated_count += 1

            return {
                'status': 'renamed',
                'old_uri': old_uri,
                'new_uri': new_uri,
                'updated_count': updated_count,
            }

        except Exception as e:
            logger.error(f'Error renaming document {old_uri} → {new_uri}: {e}')
            return {
                'status': 'error',
                'old_uri': old_uri,
                'new_uri': new_uri,
                'error': str(e),
            }

    async def sync_all(self) -> dict[str, Any]:
        """Sync all markdown documents in corpus directory.

        Returns:
            Statistics: synced, skipped, errors
        """
        synced = 0
        skipped = 0
        errors = []

        # Find all .md files in corpus
        md_files = list(self.corpus_path.rglob('*.md'))
        logger.info(f'Starting sync_all for {len(md_files)} markdown files')

        # Parallel execution with semaphore bounding
        results = await semaphore_gather(
            *[self.sync_document(file_path) for file_path in md_files]
        )

        # Aggregate results
        for result in results:
            if result['status'] == 'synced':
                synced += 1
            elif result['status'] == 'skipped':
                skipped += 1
            elif result['status'] == 'error':
                errors.append(result)

        return {
            'synced': synced,
            'skipped': skipped,
            'errors': errors,
            'total': len(md_files),
        }

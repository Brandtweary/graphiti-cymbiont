"""Document chunking utilities using Docling.

This module provides 2-stage semantic chunking using Docling's HybridChunker:
- Stage 1 (Episodes): Large chunks (~2000 tokens, 6-10 paragraphs) for LLM ingestion
- Stage 2 (Retrieval): Small chunks (~256 tokens, 1-2 paragraphs) for BM25/semantic search

Both stages use merge_peers=True to avoid tiny fragments while respecting document structure.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Stage 1: Episode ingestion (LLM context)
MAX_TOKENS_EPISODES = 2000  # Target: 6-10 paragraphs per chunk
MERGE_PEERS_EPISODES = True  # Merge paragraphs within sections

# Stage 2: Retrieval storage (search granularity)
MAX_TOKENS_RETRIEVAL = 256  # Target: 1-2 paragraphs per chunk (embedding model limit)
MERGE_PEERS_RETRIEVAL = True  # Avoid 17-token fragments, still merge short paragraphs

# Shared tokenizer
TOKENIZER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


@dataclass
class DocumentChunk:
    """Represents a single chunk from a document.

    Attributes:
        content: The text content of the chunk
        chunk_index: Position in document (0-indexed)
        total_chunks: Total number of chunks in document
        token_count: Estimated token count for this chunk
    """

    content: str
    chunk_index: int
    total_chunks: int
    token_count: int


def _reaggregate_chunks(
    fine_chunks: list[DocumentChunk],
    target_tokens: int,
    max_tokens: int,
    tokenizer,
    uri: str,
) -> list[DocumentChunk]:
    """Greedily merge small chunks up to target, with hard limit.

    Docling respects markdown structure too strictly, producing 100-150 token chunks
    even with max_tokens=2000. This reaggregates them to proper episode size.

    Args:
        fine_chunks: Small chunks from Docling (e.g., ~150 tokens each)
        target_tokens: Soft target (stop when reached, e.g., 1800)
        max_tokens: Hard limit (never exceed, e.g., 2200)
        tokenizer: For accurate token counting after merging
        uri: Document URI (for logging)

    Returns:
        Reaggregated chunks, mostly in target_tokens to max_tokens range
    """
    if not fine_chunks:
        return []

    merged_chunks = []
    current_content = []
    current_tokens = 0

    for chunk in fine_chunks:
        potential_total = current_tokens + chunk.token_count

        # Finalize if hit soft target OR next would exceed hard limit
        if current_content and (current_tokens >= target_tokens or potential_total > max_tokens):
            # Merge accumulated chunks
            merged_content = '\n\n'.join(current_content)
            # Recount tokens (joining may slightly change count)
            actual_tokens = len(tokenizer.encode(merged_content, add_special_tokens=False))

            merged_chunks.append(
                DocumentChunk(
                    content=merged_content,
                    chunk_index=len(merged_chunks),
                    total_chunks=0,  # Will update after loop
                    token_count=actual_tokens,
                )
            )

            # Start new chunk
            current_content = [chunk.content]
            current_tokens = chunk.token_count
        else:
            # Safe to add
            current_content.append(chunk.content)
            current_tokens = potential_total

    # Don't forget last chunk
    if current_content:
        merged_content = '\n\n'.join(current_content)
        actual_tokens = len(tokenizer.encode(merged_content, add_special_tokens=False))

        merged_chunks.append(
            DocumentChunk(
                content=merged_content,
                chunk_index=len(merged_chunks),
                total_chunks=0,
                token_count=actual_tokens,
            )
        )

    # Update total_chunks for all
    total = len(merged_chunks)
    for chunk in merged_chunks:
        chunk.total_chunks = total

    logger.info(
        f'Document {uri} - reaggregated {len(fine_chunks)} fine chunks into {total} episode chunks '
        f'(avg {sum(c.token_count for c in merged_chunks) / total:.0f} tokens/chunk, '
        f'target={target_tokens}, max={max_tokens})'
    )

    return merged_chunks


def chunk_document_for_episodes(file_path: str, uri: str) -> list[DocumentChunk]:
    """Chunk document for LLM episode ingestion (Stage 1: ~1800-2200 tokens).

    Creates large chunks that provide sufficient context for entity extraction and
    relationship detection during knowledge graph construction.

    Uses Docling for semantic splitting, then reaggregates to proper episode size
    since Docling respects markdown structure too strictly (produces 100-150 token chunks).

    Args:
        file_path: Absolute path to the document file
        uri: Document URI (for logging purposes)

    Returns:
        List of DocumentChunk objects optimized for LLM processing

    Note:
        Empty documents return empty list.
        Most chunks will be 1800-2200 tokens (soft target 1800, hard limit 2200).
    """
    # Get fine-grained chunks from Docling (respects document structure)
    fine_chunks = _chunk_document_internal(
        file_path, uri, MAX_TOKENS_EPISODES, MERGE_PEERS_EPISODES, stage="episode"
    )

    if not fine_chunks:
        return []

    # Reaggregate to proper episode size
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    return _reaggregate_chunks(
        fine_chunks, target_tokens=1800, max_tokens=2200, tokenizer=tokenizer, uri=uri
    )


def chunk_document_for_retrieval(file_path: str, uri: str) -> list[DocumentChunk]:
    """Chunk document for retrieval storage (Stage 2: ~256 tokens, 1-2 paragraphs).

    Creates smaller chunks optimized for BM25 keyword search and semantic similarity.
    Aligns with sentence-transformers embedding model max_seq_length of 256 tokens.

    Args:
        file_path: Absolute path to the document file
        uri: Document URI (for logging purposes)

    Returns:
        List of DocumentChunk objects optimized for retrieval

    Note:
        Empty documents return empty list.
        Single-chunk documents return one chunk with index=0.
    """
    return _chunk_document_internal(
        file_path, uri, MAX_TOKENS_RETRIEVAL, MERGE_PEERS_RETRIEVAL, stage="retrieval"
    )


def chunk_document(file_path: str, uri: str) -> list[DocumentChunk]:
    """Chunk document using episode configuration (legacy compatibility).

    DEPRECATED: Use chunk_document_for_episodes() or chunk_document_for_retrieval()
    to make the intended purpose explicit.

    This function exists for backward compatibility and defaults to episode chunking.
    """
    return chunk_document_for_episodes(file_path, uri)


def _chunk_document_internal(
    file_path: str, uri: str, max_tokens: int, merge_peers: bool, stage: str
) -> list[DocumentChunk]:
    """Internal chunking implementation used by both stage 1 and stage 2.

    Args:
        file_path: Absolute path to the document file
        uri: Document URI (for logging purposes)
        max_tokens: Maximum tokens per chunk
        merge_peers: Whether to merge adjacent chunks with same headings
        stage: "episode" or "retrieval" (for logging)

    Returns:
        List of DocumentChunk objects with metadata

    Note:
        Empty documents return empty list.
        Single-chunk documents return one chunk with index=0.
    """
    # Read file content
    path = Path(file_path)
    if not path.exists():
        logger.error(f'Document {uri} - file not found: {file_path}')
        return []

    content = path.read_text(encoding='utf-8')

    # Handle empty document
    if not content or not content.strip():
        logger.warning(f'Document {uri} - empty content, returning no chunks')
        return []

    try:
        # Initialize tokenizer for token counting
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

        # Convert document file to DoclingDocument
        converter = DocumentConverter()
        doc_result = converter.convert(source=file_path)
        dl_doc = doc_result.document

        # Initialize Docling HybridChunker with stage-specific config
        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
        )

        # Chunk the document
        chunks = list(chunker.chunk(dl_doc=dl_doc))

        # Convert Docling chunks to DocumentChunk objects
        total_chunks = len(chunks)
        document_chunks = []

        for idx, chunk in enumerate(chunks):
            # Count tokens in chunk
            tokens = tokenizer.encode(chunk.text, add_special_tokens=False)
            token_count = len(tokens)

            document_chunks.append(
                DocumentChunk(
                    content=chunk.text,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    token_count=token_count,
                )
            )

        logger.info(
            f'Document {uri} - chunked into {total_chunks} {stage} chunks '
            f'(avg {sum(c.token_count for c in document_chunks) / total_chunks:.0f} tokens/chunk, '
            f'max_tokens={max_tokens})'
        )

        return document_chunks

    except Exception as e:
        logger.error(f'Error chunking document {uri}: {e}')
        # Fallback: return whole document as single chunk
        logger.warning(f'Document {uri} - using fallback: single chunk')
        try:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
            tokens = tokenizer.encode(content, add_special_tokens=False)
            token_count = len(tokens)
        except Exception:
            # If tokenizer fails, rough estimate
            token_count = len(content.split()) * 1.3  # Rough approximation

        return [
            DocumentChunk(
                content=content,
                chunk_index=0,
                total_chunks=1,
                token_count=int(token_count),
            )
        ]

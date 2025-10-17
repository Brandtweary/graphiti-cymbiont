from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from graphiti_core.graph_queries import get_nodes_query
from graphiti_core.search.search_utils import fulltext_query

from graph_service.zep_graphiti import ZepGraphitiDep

router = APIRouter()


def aggregate_chunks_by_provenance(
    chunks: list['ChunkResult'],
    max_results: int,
) -> list['ChunkResult']:
    """Group chunks by document, restore narrative order, promote siblings.

    Args:
        chunks: Full candidate pool of ranked chunks
        max_results: Maximum number of final results

    Returns:
        Chunks ordered by (document_rank, chunk_index), sliced to max_results
    """
    # Step 1: Identify top-ranked documents (any chunk in top N)
    top_documents = {chunk.document_uri for chunk in chunks[:max_results]}

    # Step 2: Track best rank (index) per document for inter-document sorting
    doc_best_rank = {}
    for i, chunk in enumerate(chunks):
        if chunk.document_uri in top_documents:
            if chunk.document_uri not in doc_best_rank:
                doc_best_rank[chunk.document_uri] = i

    # Step 3: Collect ALL chunks from top documents (sibling promotion)
    aggregated = [chunk for chunk in chunks if chunk.document_uri in top_documents]

    # Step 4: Sort by (document_rank, chunk_index) for narrative flow
    aggregated.sort(key=lambda c: (doc_best_rank[c.document_uri], c.chunk_index))

    # Step 5: Final slice to max_results
    return aggregated[:max_results]


class ChunkResult(BaseModel):
    content: str
    document_uri: str
    chunk_index: int
    total_chunks: int
    token_count: int
    score: float


class ChunkSearchRequest(BaseModel):
    keyword_query: str = Field(description='BM25 keyword search query')
    max_results: int = Field(default=10, description='Maximum number of chunks to return')
    rerank_query: str | None = Field(
        None, description='Optional semantic reranking query using cross-encoder'
    )
    group_id: str = Field(description='Group ID for multi-tenancy')


class ChunkSearchResponse(BaseModel):
    chunks: list[ChunkResult] = Field(description='Matching chunks with metadata')
    query: str = Field(description='The processed search query')


@router.post('/search', status_code=status.HTTP_200_OK)
async def search_chunks(request: ChunkSearchRequest, graphiti: ZepGraphitiDep):
    """Search document chunks using BM25 keyword matching.

    This endpoint searches the raw chunk content for exact wording and technical precision,
    complementing the semantic entity/fact search provided by search_context.
    """
    # Build BM25 fulltext query with group_id filtering
    fuzzy_query = fulltext_query(request.keyword_query, [request.group_id], graphiti.driver)

    if fuzzy_query == '':
        return ChunkSearchResponse(chunks=[], query=request.keyword_query)

    # Construct BM25 search query
    # Over-fetch 2x to allow reranking to boost low BM25 scoring chunks
    # and ensure enough results after min_score filtering (follows Graphiti convention)
    initial_limit = request.max_results * 2

    yield_query = 'YIELD node AS c, score'
    if graphiti.driver.provider.value == 'kuzu':
        yield_query = 'WITH node AS c, score'

    query = (
        get_nodes_query(
            'chunk_content',
            '$query',
            limit=initial_limit,
            provider=graphiti.driver.provider,
        )
        + yield_query
        + """
        WITH c, score
        ORDER BY score DESC
        LIMIT $limit
        RETURN
            c.content AS content,
            c.document_uri AS document_uri,
            c.chunk_index AS chunk_index,
            c.total_chunks AS total_chunks,
            c.token_count AS token_count,
            score
        """
    )

    # Execute BM25 search
    records, _, _ = await graphiti.driver.execute_query(
        query,
        query=fuzzy_query,
        limit=initial_limit,
        routing_='r',
    )

    # Format results
    chunks = [
        ChunkResult(
            content=record['content'],
            document_uri=record['document_uri'],
            chunk_index=record['chunk_index'],
            total_chunks=record['total_chunks'],
            token_count=record['token_count'],
            score=record['score'],
        )
        for record in records
    ]

    # Semantic reranking if rerank_query provided
    if request.rerank_query:
        # Map content to chunk for reranking
        content_to_chunk_map = {chunk.content: chunk for chunk in chunks}

        # Cross-encoder reranking
        reranked_contents = await graphiti.cross_encoder.rank(
            request.rerank_query,
            list(content_to_chunk_map.keys())
        )

        # Filter by min_score and update scores (0.6 default, following Graphiti convention)
        min_score = 0.6
        reranked_chunks = []
        for content, score in reranked_contents:
            if score >= min_score:
                chunk = content_to_chunk_map[content]
                chunk.score = score
                reranked_chunks.append(chunk)

        candidate_pool = reranked_chunks
    else:
        # No reranking: use BM25 results as candidate pool
        candidate_pool = chunks

    # Provenance-based aggregation (always applied)
    chunks = aggregate_chunks_by_provenance(candidate_pool, request.max_results)

    return ChunkSearchResponse(chunks=chunks, query=request.keyword_query)

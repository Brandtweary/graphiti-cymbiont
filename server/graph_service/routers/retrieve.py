from datetime import datetime, timezone

from fastapi import APIRouter, status

from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters

from graph_service.dto import (
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    NodeSearchQuery,
    NodeSearchResults,
    SearchQuery,
    SearchResults,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    relevant_edges = await graphiti.search(
        group_ids=query.group_ids,
        query=query.query,
        num_results=query.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    return SearchResults(
        facts=facts,
    )


@router.post('/search/nodes', status_code=status.HTTP_200_OK)
async def search_nodes(query: NodeSearchQuery, graphiti: ZepGraphitiDep):
    # Configure search strategy based on center_node_uuid
    if query.center_node_uuid is not None:
        search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
    else:
        search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)

    search_config.limit = query.max_nodes

    # Build search filters
    filters = SearchFilters()
    if query.entity is not None and query.entity != '':
        filters.node_labels = [query.entity]

    # Perform search
    search_results = await graphiti.search_(
        query=query.query,
        config=search_config,
        group_ids=query.group_ids,
        center_node_uuid=query.center_node_uuid,
        search_filter=filters,
    )

    if not search_results.nodes:
        return NodeSearchResults(nodes=[])

    # Format node results
    from graph_service.dto import NodeResult

    formatted_nodes = [
        NodeResult(
            uuid=node.uuid,
            name=node.name,
            summary=node.summary if hasattr(node, 'summary') else '',
            labels=node.labels if hasattr(node, 'labels') else [],
            group_id=node.group_id,
            created_at=node.created_at,
            attributes=node.attributes if hasattr(node, 'attributes') else {},
        )
        for node in search_results.nodes
    ]

    return NodeSearchResults(nodes=formatted_nodes)


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class AddMessagesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the messages to add')
    messages: list[Message] = Field(..., description='The messages to add')


class AddEntityNodeRequest(BaseModel):
    uuid: str = Field(..., description='The uuid of the node to add')
    group_id: str = Field(..., description='The group id of the node to add')
    name: str = Field(..., description='The name of the node to add')
    summary: str = Field(default='', description='The summary of the node to add')


class AddEpisodeRequest(BaseModel):
    name: str = Field(..., description='The name of the episode')
    episode_body: str = Field(..., description='The content of the episode')
    group_id: str = Field(..., description='The group id of the episode')
    source: str = Field(default='text', description='Source type: text, message, or json')
    source_description: str = Field(default='', description='Description of the source')
    uuid: str | None = Field(default=None, description='Optional UUID for the episode')

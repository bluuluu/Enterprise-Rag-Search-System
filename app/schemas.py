from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    paths: list[str]
    visibility: str = Field(default="private", pattern="^(private|public)$")


class PermissionRequest(BaseModel):
    username: str
    can_read: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    answer: str
    citations: list[Citation]

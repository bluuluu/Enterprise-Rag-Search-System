from pydantic import BaseModel


class Settings(BaseModel):
    database_url: str = "sqlite:///./rag.db"
    embedding_dim: int = 384
    chunk_size_words: int = 120
    chunk_overlap_words: int = 25
    retrieval_top_k: int = 5
    vector_search_num_threads: int = 0
    vector_search_assume_normalized: bool = True


settings = Settings()

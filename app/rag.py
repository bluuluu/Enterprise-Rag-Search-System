import json
import time
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from .config import settings
from .embedder import Embedder
from .models import AuditLog, Chunk, Document, DocumentPermission, User
from .vector_engine import topk_cosine


embedder = Embedder(n_features=settings.embedding_dim)


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size_words - overlap_words)
    while start < len(words):
        chunk_words = words[start : start + chunk_size_words]
        chunks.append(" ".join(chunk_words))
        start += step
    return chunks


def can_read_document(db: Session, user: User, document: Document) -> bool:
    if user.role == "admin":
        return True
    if document.visibility == "public":
        return True
    if document.owner_id == user.id:
        return True

    permission = (
        db.query(DocumentPermission)
        .filter(
            and_(
                DocumentPermission.document_id == document.id,
                DocumentPermission.user_id == user.id,
                DocumentPermission.can_read.is_(True),
            )
        )
        .first()
    )
    return permission is not None


def list_accessible_documents(db: Session, user: User):
    if user.role == "admin":
        return db.query(Document).all()

    shared_ids = select(DocumentPermission.document_id).where(
        and_(
            DocumentPermission.user_id == user.id,
            DocumentPermission.can_read.is_(True),
        )
    )

    return (
        db.query(Document)
        .filter(
            or_(
                Document.visibility == "public",
                Document.owner_id == user.id,
                Document.id.in_(shared_ids),
            )
        )
        .all()
    )


def create_audit_log(
    db: Session,
    user_id: int | None,
    action: str,
    query: str | None = None,
    document_id: int | None = None,
    details: str | None = None,
    latency_ms: float | None = None,
):
    db.add(
        AuditLog(
            user_id=user_id,
            action=action,
            query=query,
            document_id=document_id,
            details=details,
            latency_ms=latency_ms,
        )
    )
    db.commit()


def index_documents(
    db: Session,
    user: User,
    paths: list[str],
    visibility: str = "private",
):
    indexed = []
    for path in paths:
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=400, detail=f"Invalid path: {path}")

        content = p.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(
            content,
            chunk_size_words=settings.chunk_size_words,
            overlap_words=settings.chunk_overlap_words,
        )
        if not chunks:
            continue

        document = Document(
            title=p.stem,
            source_path=str(p),
            owner_id=user.id,
            visibility=visibility,
        )
        db.add(document)
        db.flush()

        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            vectors = embedder.embed_batch(batch_chunks)
            for j, (text, vec) in enumerate(zip(batch_chunks, vectors, strict=True)):
                db.add(
                    Chunk(
                        document_id=document.id,
                        chunk_index=i + j,
                        text=text,
                        embedding_json=json.dumps(vec.tolist()),
                    )
                )

        create_audit_log(
            db,
            user_id=user.id,
            action="index_document",
            document_id=document.id,
            details=f"chunks={len(chunks)} path={path}",
        )
        indexed.append({"document_id": document.id, "title": document.title, "chunks": len(chunks)})
    db.commit()
    return indexed


def retrieve(db: Session, user: User, query: str, top_k: int = 5):
    t0 = time.perf_counter()
    documents = list_accessible_documents(db, user)
    if not documents:
        create_audit_log(
            db,
            user_id=user.id,
            action="search",
            query=query,
            details="no_accessible_documents",
            latency_ms=0.0,
        )
        return []

    doc_by_id = {doc.id: doc for doc in documents}
    chunks = db.query(Chunk).filter(Chunk.document_id.in_(doc_by_id.keys())).all()
    if not chunks:
        create_audit_log(
            db,
            user_id=user.id,
            action="search",
            query=query,
            details="no_chunks",
            latency_ms=0.0,
        )
        return []

    qv = embedder.embed_query(query)
    matrix = np.asarray([json.loads(chunk.embedding_json) for chunk in chunks], dtype=np.float32)
    top_indices, top_scores = topk_cosine(
        qv,
        matrix,
        top_k=top_k,
        num_threads=settings.vector_search_num_threads,
        assume_normalized=settings.vector_search_assume_normalized,
    )

    top = [
        (float(top_scores[i]), chunks[int(top_indices[i])])
        for i in range(len(top_indices))
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    create_audit_log(
        db,
        user_id=user.id,
        action="search",
        query=query,
        details=f"top_k={top_k} hits={len(top)}",
        latency_ms=latency_ms,
    )

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_title": doc_by_id[chunk.document_id].title,
            "score": round(score, 4),
            "snippet": chunk.text[:300],
        }
        for score, chunk in top
    ]


def synthesize_answer(query: str, citations: list[dict]) -> str:
    if not citations:
        return "No relevant context found for your query."

    lines = [f"Question: {query}", "Answer summary based on retrieved enterprise sources:"]
    for i, c in enumerate(citations, start=1):
        lines.append(f"{i}. {c['snippet']} [source:{c['document_title']}#{c['chunk_id']}]")
    return "\n".join(lines)

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import Base, SessionLocal, engine, get_db
from .models import Document, DocumentPermission, User
from .rag import can_read_document, index_documents, list_accessible_documents, retrieve, synthesize_answer
from .schemas import IndexRequest, PermissionRequest, SearchRequest, SearchResponse
from .security import get_current_user, require_min_role

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Enterprise RAG Knowledge Search")
DEMO_PAGE = Path(__file__).resolve().parent / "static" / "demo.html"


@app.on_event("startup")
def bootstrap_data():
    db = SessionLocal()
    users = {
        "alice_admin": "admin",
        "amy_analyst": "analyst",
        "victor_viewer": "viewer",
    }
    for username, role in users.items():
        existing = db.query(User).filter(User.username == username).first()
        if not existing:
            db.add(User(username=username, role=role))
    db.commit()
    db.close()


@app.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def demo_home():
    return FileResponse(DEMO_PAGE)


@app.get("/demo", include_in_schema=False)
def demo_page():
    return FileResponse(DEMO_PAGE)


@app.get("/users/me")
def me(user: User = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}


@app.post("/documents/index")
def index_docs(
    payload: IndexRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_min_role("analyst")),
):
    indexed = index_documents(db, user, payload.paths, visibility=payload.visibility)
    return {"indexed": indexed}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    docs = list_accessible_documents(db, user)
    return [
        {
            "id": d.id,
            "title": d.title,
            "source_path": d.source_path,
            "owner_id": d.owner_id,
            "visibility": d.visibility,
            "chunk_count": len(d.chunks),
        }
        for d in docs
    ]


@app.post("/documents/{document_id}/permissions")
def grant_permission(
    document_id: int,
    payload: PermissionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if user.role != "admin" and document.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Only owner/admin can modify permissions")

    target = db.query(User).filter(User.username == payload.username).first()
    if not target:
        raise HTTPException(status_code=404, detail="Target user not found")

    permission = (
        db.query(DocumentPermission)
        .filter(
            DocumentPermission.document_id == document_id,
            DocumentPermission.user_id == target.id,
        )
        .first()
    )
    if not permission:
        permission = DocumentPermission(document_id=document_id, user_id=target.id, can_read=payload.can_read)
        db.add(permission)
    else:
        permission.can_read = payload.can_read
    db.commit()
    return {"document_id": document_id, "username": target.username, "can_read": payload.can_read}


@app.post("/search", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_min_role("viewer")),
):
    citations = retrieve(db, user, payload.query, top_k=payload.top_k)
    answer = synthesize_answer(payload.query, citations)
    return SearchResponse(answer=answer, citations=citations)


@app.get("/documents/{document_id}/can_read")
def can_read(
    document_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"can_read": can_read_document(db, user, doc)}

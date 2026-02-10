import argparse
from dataclasses import dataclass

from app.database import SessionLocal
from app.models import User
from app.rag import retrieve


@dataclass
class EvalCase:
    query: str
    expected_doc_title: str


def evaluate(username: str, top_k: int) -> None:
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise SystemExit(f"Unknown user: {username}")

    cases = [
        EvalCase("What are our retention rules?", "retention_policy"),
        EvalCase("How does payroll approval work?", "payroll_sop"),
        EvalCase("What SSO controls are required?", "security_controls"),
    ]

    hits = 0
    for case in cases:
        citations = retrieve(db, user, case.query, top_k=top_k)
        predicted_titles = {c["document_title"] for c in citations}
        hit = case.expected_doc_title in predicted_titles
        hits += int(hit)
        print(f"query={case.query!r} expected={case.expected_doc_title!r} hit={hit}")

    recall = hits / len(cases)
    print(f"recall@{top_k}={recall:.2f} ({hits}/{len(cases)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="victor_viewer")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.username, args.top_k)

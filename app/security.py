from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from .database import get_db
from .models import User


ROLE_ORDER = {
    "viewer": 1,
    "analyst": 2,
    "admin": 3,
}


def get_current_user(
    db: Session = Depends(get_db),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> User:
    if not x_user:
        raise HTTPException(status_code=401, detail="Missing X-User header")

    user = db.query(User).filter(User.username == x_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Unknown user")
    return user


def require_min_role(min_role: str):
    def checker(user: User = Depends(get_current_user)) -> User:
        if ROLE_ORDER[user.role] < ROLE_ORDER[min_role]:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return checker

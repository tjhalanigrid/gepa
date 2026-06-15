"""Per-user settings/preferences routes."""

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session as DBSession

from ..auth import current_user
from ..db import get_db
from ..models import User, UserSettings

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
def get_settings(db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    row = db.get(UserSettings, user.id)
    return row.data if row else {}


@router.put("")
def put_settings(body: dict = Body(...), db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    row = db.get(UserSettings, user.id)
    if row:
        row.data = body
    else:
        db.add(UserSettings(user_id=user.id, data=body))
    db.commit()
    return body

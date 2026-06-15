"""Claims routes — CRUD scoped to the authenticated user."""

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from ..auth import current_user
from ..db import get_db
from ..models import Claim, User

router = APIRouter(prefix="/claims", tags=["claims"])


@router.get("")
def list_claims(db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    rows = (
        db.query(Claim)
        .filter(Claim.user_id == user.id)
        .order_by(Claim.created_at.desc())
        .all()
    )
    return [r.data for r in rows]


@router.put("/{client_id}")
def upsert_claim(client_id: str, body: dict = Body(...), db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    if not isinstance(body, dict) or not body.get("id"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Claim payload must include an id.")
    row = (
        db.query(Claim)
        .filter(Claim.user_id == user.id, Claim.client_id == client_id)
        .first()
    )
    if row:
        row.data = body
    else:
        db.add(Claim(client_id=client_id, user_id=user.id, data=body))
    db.commit()
    return body


@router.delete("/{client_id}")
def delete_claim(client_id: str, db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    db.query(Claim).filter(Claim.user_id == user.id, Claim.client_id == client_id).delete()
    db.commit()
    return {"ok": True}

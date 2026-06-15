"""Insurance claim-form routes — CRUD scoped to the authenticated user."""

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from ..auth import current_user
from ..db import get_db
from ..models import InsuranceClaim, User

router = APIRouter(prefix="/insurance", tags=["insurance"])


@router.get("")
def list_insurance(db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    rows = (
        db.query(InsuranceClaim)
        .filter(InsuranceClaim.user_id == user.id)
        .order_by(InsuranceClaim.created_at.desc())
        .all()
    )
    return [r.data for r in rows]


@router.put("/{client_id}")
def upsert_insurance(client_id: str, body: dict = Body(...), db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    if not isinstance(body, dict) or not body.get("id"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Insurance claim payload must include an id.")
    row = (
        db.query(InsuranceClaim)
        .filter(InsuranceClaim.user_id == user.id, InsuranceClaim.client_id == client_id)
        .first()
    )
    if row:
        row.data = body
    else:
        db.add(InsuranceClaim(client_id=client_id, user_id=user.id, data=body))
    db.commit()
    return body


@router.delete("/{client_id}")
def delete_insurance(client_id: str, db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    db.query(InsuranceClaim).filter(InsuranceClaim.user_id == user.id, InsuranceClaim.client_id == client_id).delete()
    db.commit()
    return {"ok": True}

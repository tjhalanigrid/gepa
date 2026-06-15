"""Vehicle registry routes — CRUD scoped to the authenticated user."""

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from ..auth import current_user
from ..db import get_db
from ..models import User, Vehicle

router = APIRouter(prefix="/vehicles", tags=["vehicles"])


@router.get("")
def list_vehicles(db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    rows = (
        db.query(Vehicle)
        .filter(Vehicle.user_id == user.id)
        .order_by(Vehicle.created_at.desc())
        .all()
    )
    return [r.data for r in rows]


@router.put("/{client_id}")
def upsert_vehicle(client_id: str, body: dict = Body(...), db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    if not isinstance(body, dict) or not body.get("id"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Vehicle payload must include an id.")
    row = (
        db.query(Vehicle)
        .filter(Vehicle.user_id == user.id, Vehicle.client_id == client_id)
        .first()
    )
    if row:
        row.data = body
    else:
        db.add(Vehicle(client_id=client_id, user_id=user.id, data=body))
    db.commit()
    return body


@router.delete("/{client_id}")
def delete_vehicle(client_id: str, db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    db.query(Vehicle).filter(Vehicle.user_id == user.id, Vehicle.client_id == client_id).delete()
    db.commit()
    return {"ok": True}

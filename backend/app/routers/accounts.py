"""Authentication routes: signup, login, logout, and current-user lookup."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from ..auth import current_user, hash_password, new_token, verify_password
from ..db import get_db
from ..models import Session, User
from ..schemas import AuthOut, LoginIn, PublicUser, SignupIn

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue_token(db: DBSession, user: User) -> str:
    token = new_token()
    db.add(Session(token=token, user_id=user.id))
    db.commit()
    return token


@router.post("/signup", response_model=AuthOut)
def signup(body: SignupIn, db: DBSession = Depends(get_db)):
    name = body.name.strip()
    phone = body.phone.strip()
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Please enter your full name.")
    if not phone:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Please enter your phone number.")
    if len(body.password) < 4:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Password must be at least 4 characters.")

    if db.query(User).filter(User.phone == phone).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this phone number already exists.")

    user = User(name=name, phone=phone, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = _issue_token(db, user)
    return AuthOut(token=token, user=PublicUser(name=user.name, phone=user.phone))


@router.post("/login", response_model=AuthOut)
def login(body: LoginIn, db: DBSession = Depends(get_db)):
    phone = body.phone.strip()
    user = db.query(User).filter(User.phone == phone).first()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No account found with this phone number.")
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect password.")
    token = _issue_token(db, user)
    return AuthOut(token=token, user=PublicUser(name=user.name, phone=user.phone))


@router.post("/logout")
def logout(authorization: str | None = None, db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    # Delete every active session for this user (simple + safe for the MVP).
    db.query(Session).filter(Session.user_id == user.id).delete()
    db.commit()
    return {"ok": True}


@router.get("/me", response_model=PublicUser)
def me(user: User = Depends(current_user)):
    return PublicUser(name=user.name, phone=user.phone)


@router.patch("/me", response_model=PublicUser)
def update_me(body: dict, db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    name = str(body.get("name", user.name)).strip()
    phone = str(body.get("phone", user.phone)).strip()
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Please enter your full name.")
    if not phone:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Please enter your phone number.")
    if phone != user.phone and db.query(User).filter(User.phone == phone).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Another account already uses this phone number.")
    user.name = name
    user.phone = phone
    db.commit()
    return PublicUser(name=user.name, phone=user.phone)


@router.post("/change-password")
def change_password(body: dict, db: DBSession = Depends(get_db), user: User = Depends(current_user)):
    current = str(body.get("currentPassword", ""))
    new = str(body.get("newPassword", ""))
    if len(new) < 4:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "New password must be at least 4 characters.")
    if not verify_password(current, user.password_hash):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current password is incorrect.")
    user.password_hash = hash_password(new)
    db.commit()
    return {"ok": True}

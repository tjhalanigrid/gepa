"""
Authentication helpers.

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib `hashlib`, no extra deps).
Sessions are opaque bearer tokens stored in the `sessions` table. The
`current_user` dependency reads `Authorization: Bearer <token>` and resolves the
account, returning 401 if the token is missing or invalid.
"""

import hashlib
import hmac
import secrets

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from .db import get_db
from .models import Session, User

_ITERATIONS = 120_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


def new_token() -> str:
    return secrets.token_urlsafe(32)


def current_user(
    authorization: str | None = Header(default=None),
    db: DBSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    session = db.get(Session, token)
    if not session:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return session.user

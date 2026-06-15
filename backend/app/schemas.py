"""Pydantic request/response models for the persistence API (auth only).

Vehicle / claim / settings payloads are passed through as raw JSON objects whose
shape is owned by the frontend TypeScript types, so they are not duplicated here.
"""

from pydantic import BaseModel


class SignupIn(BaseModel):
    name: str
    phone: str
    password: str


class LoginIn(BaseModel):
    phone: str
    password: str


class PublicUser(BaseModel):
    name: str
    phone: str


class AuthOut(BaseModel):
    token: str
    user: PublicUser

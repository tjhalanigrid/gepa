"""
Smoke tests for backend API wiring.

These verify the app imports, routers mount, and config loads — without running
any model inference. Run from the repo root: `pytest backend/tests`.
"""

from backend.app import app
from backend.app.core import config


def test_app_imports_and_has_metadata():
    assert app.title == config.API_TITLE
    assert app.version == config.API_VERSION


def test_core_routes_registered():
    paths = {getattr(r, "path", None) for r in app.routes}
    for expected in ("/", "/health", "/assess", "/job/{job_id}", "/recalculate"):
        assert expected in paths, f"missing route: {expected}"


def test_session_routes_registered():
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/session/{session_id}" in paths
    assert "/session/{session_id}/approve" in paths


def test_config_constants_present():
    assert config.ALLOWED_IMAGE_TYPES
    assert str(config.CONFIG_PATH).endswith("global_config.yaml")

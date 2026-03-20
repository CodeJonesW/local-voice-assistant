"""Tests for app.server module."""

import io
from unittest.mock import patch, MagicMock


def test_index_route():
    from app.server import app
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200


@patch("app.server._retriever")
def test_upload_route(mock_retriever, tmp_path):
    from app.server import app
    client = app.test_client()
    data = {"file": (io.BytesIO(b"hello world"), "a.txt")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200


def test_socketio_connect():
    from app.server import app, socketio
    client = socketio.test_client(app)
    assert client.is_connected()
    client.disconnect()

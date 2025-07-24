import io
from server import app, socketio, retriever


def test_upload_route(tmp_path):
    retriever.documents = []
    retriever.vectors = []
    retriever.db_path = str(tmp_path / "db.pkl")
    client = app.test_client()
    data = {"file": (io.BytesIO(b"hello world"), "a.txt")}
    resp = client.post("/upload", data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    assert retriever.retrieve("hello world")


def test_socketio_connect():
    client = socketio.test_client(app)
    assert client.is_connected()
    client.disconnect()

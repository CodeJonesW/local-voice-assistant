from retriever import Retriever


def test_add_and_retrieve(tmp_path):
    db = tmp_path / "db.pkl"
    doc = tmp_path / "doc.txt"
    doc.write_text("hello world. this is a test document.")

    r = Retriever(db_path=str(db))
    r.add_files([str(doc)])

    results = r.retrieve("test document")
    assert results
    assert any("test document" in chunk for chunk in results)

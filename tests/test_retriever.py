"""Tests for app.retriever module."""

from app.config import RetrieverConfig
from app.retriever import Retriever


def test_add_and_retrieve(tmp_path):
    db_dir = tmp_path / "chroma_db"
    doc = tmp_path / "doc.txt"
    doc.write_text("hello world. this is a test document about machine learning.")

    config = RetrieverConfig(db_path=str(db_dir))
    r = Retriever(config)
    r.add_files([str(doc)])

    results = r.retrieve("machine learning")
    assert results
    assert any("machine learning" in chunk for chunk in results)


def test_add_folder_moves_files(tmp_path):
    db_dir = tmp_path / "chroma_db"
    folder = tmp_path / "toTrain"
    processed = tmp_path / "processed"
    folder.mkdir()

    f1 = folder / "a.txt"
    f2 = folder / "b.txt"
    f1.write_text("hello world document one")
    f2.write_text("goodbye world document two")

    config = RetrieverConfig(
        db_path=str(db_dir),
        train_folder=str(folder),
        processed_folder=str(processed),
    )
    r = Retriever(config)
    r.add_folder()

    # Files should be moved
    assert not f1.exists()
    assert not f2.exists()
    assert (processed / "a.txt").exists()
    assert (processed / "b.txt").exists()

    # Documents should be retrievable
    results = r.retrieve("hello")
    assert results


def test_retrieve_empty(tmp_path):
    db_dir = tmp_path / "chroma_db"
    config = RetrieverConfig(db_path=str(db_dir))
    r = Retriever(config)
    results = r.retrieve("anything")
    assert results == []

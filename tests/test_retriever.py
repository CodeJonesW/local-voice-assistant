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


def test_add_folder_moves_files(tmp_path):
    db = tmp_path / "db.pkl"
    folder = tmp_path / "toTrain"
    processed = tmp_path / "previouslyTrainedOn"
    folder.mkdir()
    processed.mkdir()

    f1 = folder / "a.txt"
    f2 = folder / "b.txt"
    f1.write_text("hello")
    f2.write_text("world")

    r = Retriever(db_path=str(db))
    r.add_folder(str(folder), processed_folder=str(processed))

    # Files should be moved
    assert not f1.exists() and not f2.exists()
    assert (processed / "a.txt").exists()
    assert (processed / "b.txt").exists()

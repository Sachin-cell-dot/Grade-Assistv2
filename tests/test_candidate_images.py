from tools.vision import list_candidate_images


def test_lists_only_image_candidates_newest_first(tmp_path):
    old = tmp_path / "old.jpg"
    newest = tmp_path / "new.png"
    ignored = tmp_path / "notes.txt"
    old.write_bytes(b"old")
    newest.write_bytes(b"new")
    ignored.write_text("not an image")
    assert list_candidate_images(tmp_path) == [newest, old]

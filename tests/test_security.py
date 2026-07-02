"""Security regression tests for the topology generator.

Covers the path-traversal fix: results_id / download routes must reject anything that
isn't a plain UUID so a segment like '..' can't escape RESULTS_FOLDER.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as topo  # noqa: E402


def test_valid_results_id_rejects_traversal_and_junk():
    for bad in ["..", "../../etc/passwd", "foo", "", "..%2f", "a/b"]:
        assert topo._valid_results_id(bad) is False, bad
    assert topo._valid_results_id(str(uuid.uuid4())) is True


def _client():
    topo.app.config["TESTING"] = True
    return topo.app.test_client()


def test_download_zip_rejects_invalid_id():
    r = _client().get("/download_zip/nope")
    assert r.status_code in (301, 302)                      # redirected to index, not served
    assert "application/zip" not in (r.headers.get("Content-Type") or "")


def test_results_route_rejects_invalid_id():
    r = _client().get("/results/nope")
    assert r.status_code in (301, 302)


def test_download_file_rejects_invalid_id():
    r = _client().get("/download/nope/whatever.txt")
    assert r.status_code == 404


def test_wellformed_uuid_passes_guard_then_not_found():
    # A valid but nonexistent id must pass validation and hit the normal not-found
    # path (a redirect), i.e. valid ids are not blocked and no traversal/500 occurs.
    r = _client().get("/download_zip/" + str(uuid.uuid4()))
    assert r.status_code in (301, 302)

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_connector_publish.py"
SPEC = importlib.util.spec_from_file_location("prepare_connector_publish", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_bytes,
        capture_output=True,
        check=True,
    )
    return result.stdout


def make_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test Agent")
    git(repo, "config", "user.email", "test@example.invalid")

    (repo / "keep.txt").write_text("base\n", encoding="utf-8")
    (repo / "delete.txt").write_text("remove me\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    base = git(repo, "rev-parse", "HEAD").decode().strip()

    (repo / "keep.txt").write_text("line one\r\nline two\r\n", encoding="utf-8", newline="")
    (repo / "binary.dat").write_bytes(b"\x00\xa1\xff\r\n")
    (repo / "delete.txt").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "head")
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    return repo, base, head


def test_plan_uses_exact_committed_blob_bytes(tmp_path: Path) -> None:
    repo, base, head = make_repo(tmp_path)
    output = tmp_path / "plan"
    base_tree = git(repo, "rev-parse", f"{base}^{{tree}}").decode().strip()

    manifest = MODULE.build_publish_plan(
        repo=repo,
        base_ref=base,
        head_ref=head,
        repository_full_name="owner/repo",
        remote_base_commit="a" * 40,
        branch_name="agent/test",
        output_dir=output,
        expected_base_tree=base_tree,
    )

    assert manifest["expected_tree"] == git(repo, "rev-parse", f"{head}^{{tree}}").decode().strip()
    changed = {entry["path"]: entry for entry in manifest["changed_files"]}
    assert set(changed) == {"binary.dat", "delete.txt", "keep.txt"}
    assert changed["delete.txt"]["blob_sha"] is None

    for path in ("binary.dat", "keep.txt"):
        entry = changed[path]
        payload = json.loads((output / entry["payload_file"]).read_text(encoding="utf-8"))
        decoded = base64.b64decode(payload["content"])
        committed = git(repo, "cat-file", "blob", entry["blob_sha"])
        assert decoded == committed

    tree_payload = json.loads((output / "create-tree.json").read_text(encoding="utf-8"))
    assert tree_payload["base_tree_sha"] == base_tree
    assert any(item == {"path": "delete.txt", "sha": None} for item in tree_payload["tree_elements"])


def test_plan_rejects_dirty_worktree(tmp_path: Path) -> None:
    repo, base, head = make_repo(tmp_path)
    (repo / "uncommitted.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(MODULE.PublishPlanError, match="not clean"):
        MODULE.build_publish_plan(
            repo=repo,
            base_ref=base,
            head_ref=head,
            repository_full_name="owner/repo",
            remote_base_commit="a" * 40,
            branch_name="agent/test",
            output_dir=tmp_path / "plan",
        )


def test_plan_rejects_remote_tree_mismatch(tmp_path: Path) -> None:
    repo, base, head = make_repo(tmp_path)

    with pytest.raises(MODULE.PublishPlanError, match="does not match"):
        MODULE.build_publish_plan(
            repo=repo,
            base_ref=base,
            head_ref=head,
            repository_full_name="owner/repo",
            remote_base_commit="a" * 40,
            branch_name="agent/test",
            output_dir=tmp_path / "plan",
            expected_base_tree="b" * 40,
        )

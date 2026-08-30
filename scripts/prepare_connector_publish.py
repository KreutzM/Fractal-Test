"""Prepare exact Git-data payloads for connector-only publication.

The script reads committed Git objects instead of working-tree text. This keeps
line endings, encodings, executable modes, and blob SHAs identical between the
validated local commit and the GitHub Git-data API publication.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class PublishPlanError(RuntimeError):
    """Raised when a safe publish plan cannot be produced."""


@dataclass(frozen=True, slots=True)
class ChangedPath:
    status: str
    path: str
    old_path: str | None = None


def _git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    command = ["git", "-C", str(repo), *args]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise PublishPlanError(f"{' '.join(command)} failed: {stderr}")
    if text:
        return result.stdout.decode("utf-8", errors="strict").strip()
    return result.stdout


def _require_clean_worktree(repo: Path) -> None:
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise PublishPlanError(
            "working tree is not clean; commit or remove all changes before "
            "preparing connector payloads"
        )


def _parse_changed_paths(raw: bytes) -> tuple[ChangedPath, ...]:
    tokens = raw.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()

    changes: list[ChangedPath] = []
    index = 0
    while index < len(tokens):
        status = tokens[index].decode("ascii")
        index += 1
        code = status[0]
        if code in {"R", "C"}:
            if index + 1 >= len(tokens):
                raise PublishPlanError("incomplete rename/copy record from git diff")
            old_path = tokens[index].decode("utf-8", errors="surrogateescape")
            new_path = tokens[index + 1].decode("utf-8", errors="surrogateescape")
            index += 2
            changes.append(ChangedPath(status=status, path=new_path, old_path=old_path))
        else:
            if index >= len(tokens):
                raise PublishPlanError("incomplete path record from git diff")
            path = tokens[index].decode("utf-8", errors="surrogateescape")
            index += 1
            changes.append(ChangedPath(status=status, path=path))
    return tuple(changes)


def _changed_paths(repo: Path, base_ref: str, head_ref: str) -> tuple[ChangedPath, ...]:
    raw = _git(
        repo,
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        base_ref,
        head_ref,
        text=False,
    )
    assert isinstance(raw, bytes)
    return _parse_changed_paths(raw)


def _tree_entry(repo: Path, ref: str, path: str) -> tuple[str, str, str]:
    raw = _git(repo, "ls-tree", "-z", ref, "--", path, text=False)
    assert isinstance(raw, bytes)
    if not raw:
        raise PublishPlanError(f"path {path!r} does not exist in {ref}")
    header, stored_path = raw.rstrip(b"\0").split(b"\t", 1)
    if stored_path.decode("utf-8", errors="surrogateescape") != path:
        raise PublishPlanError(f"git returned an unexpected path for {path!r}")
    mode, object_type, sha = header.decode("ascii").split()
    return mode, object_type, sha


def _blob_bytes(repo: Path, sha: str) -> bytes:
    raw = _git(repo, "cat-file", "blob", sha, text=False)
    assert isinstance(raw, bytes)
    return raw


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_publish_plan(
    *,
    repo: Path,
    base_ref: str,
    head_ref: str,
    repository_full_name: str,
    remote_base_commit: str,
    branch_name: str,
    output_dir: Path,
    expected_base_tree: str | None = None,
) -> dict[str, object]:
    repo = repo.resolve()
    output_dir = output_dir.resolve()
    _require_clean_worktree(repo)

    _git(repo, "rev-parse", "--verify", f"{base_ref}^{{commit}}")
    _git(repo, "rev-parse", "--verify", f"{head_ref}^{{commit}}")
    try:
        _git(repo, "merge-base", "--is-ancestor", base_ref, head_ref)
    except PublishPlanError as exc:
        raise PublishPlanError(f"{base_ref} is not an ancestor of {head_ref}") from exc

    local_base_commit = str(_git(repo, "rev-parse", f"{base_ref}^{{commit}}"))
    local_base_tree = str(_git(repo, "rev-parse", f"{base_ref}^{{tree}}"))
    head_commit = str(_git(repo, "rev-parse", f"{head_ref}^{{commit}}"))
    expected_tree = str(_git(repo, "rev-parse", f"{head_ref}^{{tree}}"))

    if expected_base_tree is not None and expected_base_tree != local_base_tree:
        raise PublishPlanError(
            "remote target tree does not match the verified local base tree: "
            f"remote={expected_base_tree}, local={local_base_tree}"
        )

    changes = _changed_paths(repo, base_ref, head_ref)
    if not changes:
        raise PublishPlanError("no committed changes exist between base and head")

    output_dir.mkdir(parents=True, exist_ok=True)
    blob_dir = output_dir / "blobs"
    blob_dir.mkdir(exist_ok=True)

    blob_uploads: list[dict[str, object]] = []
    tree_elements: list[dict[str, object]] = []
    changed_files: list[dict[str, object]] = []
    written_blob_shas: set[str] = set()

    for change in changes:
        code = change.status[0]
        if change.old_path is not None and change.old_path != change.path:
            tree_elements.append({"path": change.old_path, "sha": None})

        if code == "D":
            tree_elements.append({"path": change.path, "sha": None})
            changed_files.append(
                {
                    "status": change.status,
                    "path": change.path,
                    "old_path": change.old_path,
                    "blob_sha": None,
                    "size_bytes": 0,
                }
            )
            continue

        mode, object_type, sha = _tree_entry(repo, head_ref, change.path)
        if object_type != "blob":
            raise PublishPlanError(
                f"unsupported Git object type {object_type!r} for {change.path!r}"
            )
        content = _blob_bytes(repo, sha)
        payload_file = f"blobs/{sha}.json"
        if sha not in written_blob_shas:
            payload = {
                "repository_full_name": repository_full_name,
                "content": base64.b64encode(content).decode("ascii"),
                "encoding": "base64",
            }
            _json_write(output_dir / payload_file, payload)
            blob_uploads.append(
                {
                    "sha": sha,
                    "size_bytes": len(content),
                    "payload_file": payload_file,
                }
            )
            written_blob_shas.add(sha)

        tree_elements.append(
            {"path": change.path, "mode": mode, "type": "blob", "sha": sha}
        )
        changed_files.append(
            {
                "status": change.status,
                "path": change.path,
                "old_path": change.old_path,
                "mode": mode,
                "blob_sha": sha,
                "size_bytes": len(content),
                "payload_file": payload_file,
            }
        )

    base_tree_for_publish = expected_base_tree or local_base_tree
    create_tree_payload = {
        "repository_full_name": repository_full_name,
        "base_tree_sha": base_tree_for_publish,
        "tree_elements": tree_elements,
    }
    create_commit_template = {
        "repository_full_name": repository_full_name,
        "message": str(_git(repo, "log", "-1", "--format=%s", head_ref)),
        "tree_sha": "<returned-tree-sha>",
        "parent_sha": remote_base_commit,
    }
    create_branch_template = {
        "repository_full_name": repository_full_name,
        "branch_name": branch_name,
        "sha": "<returned-commit-sha>",
    }
    compare_payload = {
        "repo_full_name": repository_full_name,
        "base": remote_base_commit,
        "head": branch_name,
    }

    _json_write(output_dir / "create-tree.json", create_tree_payload)
    _json_write(output_dir / "create-commit-template.json", create_commit_template)
    _json_write(output_dir / "create-branch-template.json", create_branch_template)
    _json_write(output_dir / "compare.json", compare_payload)

    manifest: dict[str, object] = {
        "format_version": 1,
        "repository_full_name": repository_full_name,
        "branch_name": branch_name,
        "remote_base_commit": remote_base_commit,
        "local_base_ref": base_ref,
        "local_base_commit": local_base_commit,
        "local_base_tree": local_base_tree,
        "head_ref": head_ref,
        "head_commit": head_commit,
        "expected_tree": expected_tree,
        "changed_files": changed_files,
        "blob_uploads": blob_uploads,
        "create_tree_payload": "create-tree.json",
        "create_commit_template": "create-commit-template.json",
        "create_branch_template": "create-branch-template.json",
        "compare_payload": "compare.json",
    }
    _json_write(output_dir / "manifest.json", manifest)

    instructions = f"""Connector publish plan
======================

Repository: {repository_full_name}
Remote parent: {remote_base_commit}
Local base tree: {local_base_tree}
Expected final tree: {expected_tree}
Feature branch: {branch_name}
Changed paths: {len(changed_files)}
Unique blobs: {len(blob_uploads)}

Required sequence
-----------------
1. Create each blob using blobs/*.json. Every returned SHA must equal the SHA in manifest.json.
2. Create the tree using create-tree.json. The returned SHA must equal {expected_tree}.
3. Create the commit using create-commit-template.json after replacing <returned-tree-sha>.
4. Create the branch using create-branch-template.json after replacing <returned-commit-sha>.
5. Compare using compare.json and require behind_by == 0 plus only the expected paths.
6. Open a draft pull request.

Stop immediately on the first SHA mismatch. Do not retry by copying or re-encoding file text.
"""
    (output_dir / "README.txt").write_text(instructions, encoding="utf-8", newline="\n")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare exact GitHub Git-data connector payloads from a committed diff."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--repository", required=True, dest="repository_full_name")
    parser.add_argument("--remote-base-commit", required=True)
    parser.add_argument("--branch", required=True, dest="branch_name")
    parser.add_argument("--expected-base-tree")
    parser.add_argument("--output-dir", type=Path, default=Path(".agent-publish"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_publish_plan(
            repo=args.repo,
            base_ref=args.base_ref,
            head_ref=args.head_ref,
            repository_full_name=args.repository_full_name,
            remote_base_commit=args.remote_base_commit,
            branch_name=args.branch_name,
            output_dir=args.output_dir,
            expected_base_tree=args.expected_base_tree,
        )
    except PublishPlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Prepared {len(manifest['changed_files'])} changed paths in {args.output_dir}")
    print(f"Expected final tree: {manifest['expected_tree']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

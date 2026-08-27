"""Fetch source-only snapshots of the baseline authors' GitHub repositories.

The normal ``git clone`` transport is unreliable on the current network.  This
script uses the official GitHub REST API and raw.githubusercontent.com instead,
records the exact commit, and preserves every upstream relative source path.

Only source/config/documentation files up to ``--max-bytes`` are downloaded;
datasets, checkpoints, PDFs, and generated outputs are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


USER_AGENT = "FCPC-baseline-source-fetcher/1.0"
SOURCE_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
}
SOURCE_NAMES = {
    ".gitignore",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "README",
    "requirements.txt",
}


@dataclass(frozen=True)
class Repository:
    name: str
    owner: str
    repo: str
    branch: str
    redistributable: bool
    note: str = ""

    @property
    def upstream_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"


REPOSITORIES = (
    Repository(
        "FedProx",
        "litian96",
        "FedProx",
        "master",
        True,
        "Official MLSys 2020 implementation; core trainer is flearn/trainers/fedprox.py.",
    ),
    Repository(
        "MOON",
        "Xtra-Computing",
        "MOON",
        "main",
        True,
        "Official CVPR 2021 implementation; training entry is main.py.",
    ),
    Repository(
        "FedDyn",
        "alpemreacar",
        "FedDyn",
        "master",
        True,
        "Official ICLR 2021 dynamic-regularization implementation; methods are in utils_methods.py.",
    ),
    Repository(
        "FBLG",
        "YingLi-Y",
        "FBLG",
        "master",
        False,
        "Official IJCAI 2024 code; no repository license was found, so do not redistribute the snapshot.",
    ),
    Repository(
        "FedCFA",
        "hua-zi",
        "FedCFA",
        "main",
        True,
        "Official AAAI 2025 implementation; algorithm entry is alg/fedcfa.py.",
    ),
)


def _open_with_retry(url: str, attempts: int = 6):
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            return urlopen(request, timeout=60)
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            last_error = error
        except OSError as error:
            last_error = error
        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 16))
    assert last_error is not None
    raise last_error


def _read_json(url: str):
    with _open_with_retry(url) as response:
        return json.load(response)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    with _open_with_retry(url) as response:
        destination.write_bytes(response.read())


def _is_source(path: str, size: int | None, max_bytes: int) -> bool:
    item = Path(path)
    if size is not None and size > max_bytes:
        return False
    return item.name in SOURCE_NAMES or item.suffix.lower() in SOURCE_SUFFIXES


def fetch_repository(spec: Repository, output_root: Path, max_bytes: int) -> dict:
    commit_api = (
        f"https://api.github.com/repos/{spec.owner}/{spec.repo}/commits/"
        f"{quote(spec.branch, safe='')}"
    )
    commit = _read_json(commit_api)["sha"]
    tree_api = (
        f"https://api.github.com/repos/{spec.owner}/{spec.repo}/git/trees/"
        f"{commit}?recursive=1"
    )
    tree = _read_json(tree_api)["tree"]
    selected = [
        node
        for node in tree
        if node.get("type") == "blob"
        and _is_source(node["path"], node.get("size"), max_bytes)
    ]

    repository_root = output_root / spec.name
    downloaded: list[str] = []
    for node in selected:
        relative_path = node["path"]
        raw_url = (
            f"https://raw.githubusercontent.com/{spec.owner}/{spec.repo}/"
            f"{commit}/{quote(relative_path, safe='/')}"
        )
        _download(raw_url, repository_root / relative_path)
        downloaded.append(relative_path)

    manifest = {
        **asdict(spec),
        "upstream_url": spec.upstream_url,
        "commit": commit,
        "source_file_count": len(downloaded),
        "source_files": downloaded,
    }
    (repository_root / "FCPC_UPSTREAM_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _select_repositories(names: Iterable[str]) -> tuple[Repository, ...]:
    requested = {name.lower() for name in names}
    if not requested:
        return REPOSITORIES
    selected = tuple(spec for spec in REPOSITORIES if spec.name.lower() in requested)
    missing = requested - {spec.name.lower() for spec in selected}
    if missing:
        raise ValueError(f"unknown repositories: {', '.join(sorted(missing))}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("third_party/baselines"),
        help="snapshot root (default: third_party/baselines)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=2 * 1024 * 1024,
        help="maximum size for one source file",
    )
    parser.add_argument("names", nargs="*", help="optional repository names")
    parser.add_argument(
        "--include-unlicensed",
        action="store_true",
        help="also fetch repositories without a redistribution license",
    )
    args = parser.parse_args()

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = _select_repositories(args.names)
    if not args.include_unlicensed:
        selected = tuple(spec for spec in selected if spec.redistributable)
    for spec in selected:
        manifest = fetch_repository(spec, output_root, args.max_bytes)
        print(
            f"{spec.name}: {manifest['commit']} "
            f"({manifest['source_file_count']} source files)"
        )


if __name__ == "__main__":
    main()

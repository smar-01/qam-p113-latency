"""Pack and safely restore the project's private research release archives.

Only the Python standard library is required. Archives contain data/ and out/;
credentials and the Python environment are never selected by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
from datetime import datetime, timezone
import zipfile


BUFFER = 8 * 1024 * 1024
MAX_ASSET_BYTES = 2 * 1024**3
STORED_SUFFIXES = {".zst", ".gz", ".png", ".jpg", ".jpeg", ".zip", ".parquet"}
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BUFFER):
            digest.update(block)
    return digest.hexdigest()


def safe_member(value: str) -> str:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid archive member path: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value
            or len(path.parts) < 2 or path.parts[0] not in {"data", "out"}
            or any(part in {"", ".", ".."} or ":" in part for part in path.parts)):
        raise ValueError(f"Unsafe archive member path: {value!r}")
    return value


def check_no_symlink(path: Path, root: Path) -> None:
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Resolved path is outside the destination: {path}")
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {current}")
        if current == root:
            return
        if current.parent == current:
            raise ValueError(f"Path is outside the destination: {path}")
        current = current.parent


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def pack(root: Path, output: Path, manifest_path: Path, chunk_bytes: int) -> dict:
    root = root.resolve()
    output = output.resolve()
    if chunk_bytes <= 0 or chunk_bytes > int(1.75 * 1024**3):
        raise ValueError("Chunk size must be positive and no more than 1792 MiB")
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists: {manifest_path}")
    files: list[Path] = []
    for folder in (root / "data", root / "out"):
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"Expected a real directory: {folder}")
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Refusing to archive symlink: {path}")
            if path.is_file():
                files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError("No research files found")
    chunks: list[list[Path]] = []
    current: list[Path] = []
    current_size = 0
    for path in files:
        size = path.stat().st_size
        if size > chunk_bytes:
            raise ValueError(f"File exceeds chunk target: {path}")
        if current and current_size + size > chunk_bytes:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(path)
        current_size += size
    if current:
        chunks.append(current)
    output.mkdir(parents=True, exist_ok=True)
    names = [f"research-{i:03d}.zip" for i in range(1, len(chunks) + 1)]
    for name in names:
        if (output / name).exists():
            raise FileExistsError(f"Archive already exists: {output / name}")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": ["data/", "out/"],
        "description": "Complete research data and output snapshot; credentials and runtime environments excluded.",
        "file_count": len(files),
        "original_bytes": 0,
        "archives": [],
        "files": [],
    }
    print(f"Packing {len(files)} files into {len(chunks)} independent ZIP archives", flush=True)
    for index, (name, paths) in enumerate(zip(names, chunks), start=1):
        archive_path = output / name
        with zipfile.ZipFile(archive_path, "x", allowZip64=True) as archive:
            for path in paths:
                member = safe_member(path.relative_to(root).as_posix())
                before = path.stat()
                info = zipfile.ZipInfo.from_file(path, arcname=member)
                info.compress_type = (zipfile.ZIP_STORED if path.suffix.lower() in STORED_SUFFIXES
                                      else zipfile.ZIP_DEFLATED)
                info._compresslevel = 1
                digest = hashlib.sha256()
                copied = 0
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    while block := source.read(BUFFER):
                        target.write(block)
                        digest.update(block)
                        copied += len(block)
                after = path.stat()
                if copied != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Source changed while archiving: {member}")
                manifest["original_bytes"] += copied
                manifest["files"].append({"path": member, "bytes": copied,
                                          "sha256": digest.hexdigest(), "archive": name})
        archive_size = archive_path.stat().st_size
        if archive_size >= MAX_ASSET_BYTES:
            raise RuntimeError(f"Archive exceeds GitHub asset limit: {name}")
        with zipfile.ZipFile(archive_path) as archive:
            bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC verification failed: {name}: {bad}")
        manifest["archives"].append({"name": name, "bytes": archive_size,
                                     "sha256": digest_file(archive_path), "file_count": len(paths)})
        print(f"Verified {index}/{len(chunks)}: {name}, {len(paths)} files, {archive_size / 1e9:.3f} GB", flush=True)
    manifest["archive_bytes"] = sum(row["bytes"] for row in manifest["archives"])
    write_json(manifest_path, manifest)
    copy_path = output / manifest_path.name
    if copy_path.resolve() != manifest_path.resolve():
        write_json(copy_path, manifest)
    print(f"Complete: {manifest['file_count']} files, {manifest['original_bytes']} original bytes, "
          f"{manifest['archive_bytes']} archive bytes", flush=True)
    return manifest


def load_manifest(path: Path) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported research archive manifest version")
    archives = {}
    for row in manifest["archives"]:
        name = row["name"]
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Unsafe archive filename")
        if name in archives:
            raise ValueError("Duplicate archive filename")
        archives[name] = row
    files = {}
    for row in manifest["files"]:
        member = safe_member(row["path"])
        if member in files or row["archive"] not in archives:
            raise ValueError("Duplicate member or unknown archive")
        files[member] = row
    if len(files) != manifest["file_count"] or sum(row["bytes"] for row in files.values()) != manifest["original_bytes"]:
        raise ValueError("Manifest totals do not match")
    return manifest, archives, files


def validate_archives(archive_dir: Path, archives: dict, files: dict) -> None:
    for name, row in archives.items():
        path = archive_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Archive is missing or has the wrong size: {name}")
        if digest_file(path) != row["sha256"]:
            raise ValueError(f"Archive SHA-256 mismatch: {name}")
        expected = {member for member, entry in files.items() if entry["archive"] == name}
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            actual = [safe_member(info.filename) for info in infos]
            if len(actual) != len(set(actual)) or set(actual) != expected or len(actual) != row["file_count"]:
                raise ValueError(f"Archive member allowlist mismatch: {name}")
            for info in infos:
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                    raise ValueError(f"Unsupported archive member: {info.filename}")
                if info.file_size != files[info.filename]["bytes"]:
                    raise ValueError(f"Wrong original file size: {info.filename}")
        print(f"Validated archive SHA-256 and member list: {name}", flush=True)


def restore(manifest_path: Path, archive_dir: Path, destination: Path, verify_only: bool = False) -> None:
    manifest, archives, files = load_manifest(manifest_path)
    validate_archives(archive_dir.resolve(), archives, files)
    destination = destination.absolute()
    if destination.is_symlink():
        raise ValueError("Destination cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    reused = 0
    restored = 0
    for name in archives:
        with zipfile.ZipFile(archive_dir / name) as archive:
            for info in archive.infolist():
                row = files[info.filename]
                target = destination.joinpath(*PurePosixPath(info.filename).parts)
                check_no_symlink(target, destination)
                if target.exists():
                    if not target.is_file() or target.stat().st_size != row["bytes"] or digest_file(target) != row["sha256"]:
                        raise FileExistsError(f"Refusing to overwrite different existing file: {target}")
                    reused += 1
                    continue
                if verify_only:
                    raise FileNotFoundError(f"Restored file is missing: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                check_no_symlink(target.parent, destination)
                digest = hashlib.sha256()
                copied = 0
                # Exclusive creation prevents replacement of any existing research file.
                try:
                    stream = target.open("xb")
                except FileExistsError:
                    raise FileExistsError(f"Destination appeared during restore: {target}") from None
                try:
                    with stream, archive.open(info) as source:
                        while block := source.read(BUFFER):
                            stream.write(block)
                            digest.update(block)
                            copied += len(block)
                    if copied != row["bytes"] or digest.hexdigest() != row["sha256"]:
                        raise ValueError(f"Restored file SHA-256 mismatch: {info.filename}")
                except BaseException:
                    target.unlink(missing_ok=True)
                    raise
                restored += 1
        print(f"Checked/restored members: {name}", flush=True)
    print(f"Complete: {manifest['file_count']} files; {restored} restored, {reused} existing files verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack_parser = commands.add_parser("pack", help="Create a complete data/ and out/ snapshot")
    pack_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    pack_parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / ".tmp/cloud-upload")
    pack_parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "research-archive-manifest.json")
    pack_parser.add_argument("--chunk-mib", type=int, default=1536)
    for command in ("restore", "verify"):
        sub = commands.add_parser(command, help=f"{command.title()} research files using their SHA-256 manifest")
        sub.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "research-archive-manifest.json")
        sub.add_argument("--archives", type=Path, required=True)
        sub.add_argument("--destination", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.root, args.output, args.manifest, args.chunk_mib * 1024**2)
    else:
        restore(args.manifest, args.archives, args.destination, verify_only=args.command == "verify")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"Archive operation failed: {error}", file=sys.stderr)
        raise SystemExit(1)

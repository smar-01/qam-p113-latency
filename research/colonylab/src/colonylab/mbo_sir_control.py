"""Stop one verified logical fetch and preserve its unconfirmed prefixes.

This transport controller never changes the frozen requests, completed stream
markers or receipts. Process identity requires the exact workspace, Python
script and --fetch-only mode. A Windows venv redirector/worker may form one
verified linear chain; only its deepest worker is terminated. Every matching
parent must exit normally before any filesystem mutation.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4


_PYTHON_NAME = re.compile(r"python(?:w|\d+(?:\.\d+)*)?(?:\.exe)?", re.IGNORECASE)
_SIMPLE_OPTIONS = {"-u", "-B", "-E", "-s", "-S", "-I", "-O", "-OO", "-q", "-v", "-vv", "-d", "-b", "-bb"}
_STOP_SECONDS = 20.


class FetchControlError(RuntimeError):
    """Fail-closed transport control error; messages never include command lines."""


def _inside(path: Path, parent: Path, *, allow_equal: bool = False) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(parent) or (resolved == parent and not allow_equal):
        raise FetchControlError("Configured transport path escapes its verified workspace directory.")
    return resolved


def _absolute(root: Path, value: str) -> Path:
    path = Path(value)
    return _inside(path if path.is_absolute() else root / path, root)


def _resolve_workspace(root: Path) -> Path:
    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir():
        raise FetchControlError("Workspace root must be an existing directory.")
    return resolved


def _script_and_args(command: list[str]) -> tuple[str, list[str]] | None:
    if not command:
        return None
    index = 1
    while index < len(command):
        value = command[index]
        if value in {"-c", "-m", "-"}:
            return None
        if value == "--":
            index += 1
            break
        if value in _SIMPLE_OPTIONS:
            index += 1
        elif value in {"-W", "-X", "--check-hash-based-pycs"}:
            index += 2
        elif (value.startswith("-W") or value.startswith("-X")) and len(value) > 2:
            index += 1
        elif value.startswith("-"):
            return None
        else:
            break
    return (command[index], command[index + 1:]) if index < len(command) else None


def _matches(process, root: Path) -> bool:
    if int(process.pid) == os.getpid() or not _PYTHON_NAME.fullmatch(process.name()):
        return False
    # Read another process's command line only after its cwd is verified.
    raw_cwd = process.cwd()
    if not isinstance(raw_cwd, str) or not raw_cwd.strip() or not Path(raw_cwd).is_absolute():
        raise FetchControlError("Process working directory is not an explicit absolute path; no action taken.")
    cwd = Path(raw_cwd).resolve()
    if cwd != root:
        return False
    parsed = _script_and_args(list(process.cmdline()))
    if parsed is None:
        return False
    script, args = parsed
    candidate = Path(script)
    candidate = (candidate if candidate.is_absolute() else cwd / candidate).resolve()
    if candidate != (root / "scripts/run_mbo_sir.py").resolve():
        return False
    return "--fetch-only" in args and not set(args).intersection(
        {"--analyze-only", "--replay-available", "--watch-replay", "--stop-fetch"})


def _chain(matches: list) -> list[tuple[object, float, int]]:
    """Require one nonbranching, creation-time-consistent parent chain."""
    if not matches:
        raise FetchControlError("No exact running fetch matches this workspace; no action taken.")
    identities = {}
    for process in matches:
        created = float(process.create_time())
        if not math.isfinite(created) or created <= 0:
            raise FetchControlError("Fetch creation time could not be verified; no action taken.")
        if process.pid in identities:
            raise FetchControlError("Duplicate process identity; no action taken.")
        identities[process.pid] = (process, created, int(process.ppid()))
    roots = [pid for pid, (_, _, parent) in identities.items() if parent not in identities]
    children = {pid: [] for pid in identities}
    for pid, (_, created, parent) in identities.items():
        if parent in identities:
            if created < identities[parent][1]:
                raise FetchControlError("Process parent/child start times are inconsistent; no action taken.")
            children[parent].append(pid)
    if len(roots) != 1 or any(len(value) > 1 for value in children.values()):
        raise FetchControlError("Multiple unrelated or branching fetches match; no action taken.")
    ordered, pid = [], roots[0]
    while True:
        ordered.append(identities[pid])
        if not children[pid]:
            break
        pid = children[pid][0]
        if len(ordered) >= len(identities):
            raise FetchControlError("Cyclic process identities; no action taken.")
    if len(ordered) != len(identities):
        raise FetchControlError("Unrelated fetch process identities; no action taken.")
    return ordered


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def _get_process_api():
    if importlib.util.find_spec("psutil") is not None:
        import psutil
        return psutil
    return _WindowsProcessAPI()


def _own_ancestry(api) -> set[int]:
    """Exclude this controller and every ancestor, including a venv launcher."""
    excluded = set()
    own_pid = os.getpid()
    current = own_pid
    for depth in range(128):
        if current <= 0:
            return excluded
        if current in excluded:
            raise FetchControlError("Controller ancestry is cyclic; no action taken.")
        excluded.add(current)
        try:
            current = int(api.Process(current).ppid()) if hasattr(api, "Process") else int(api.parent_pid(current))
        except Exception as exc:
            missing = isinstance(exc, ProcessLookupError) or type(exc).__name__ in {"NoSuchProcess", "ZombieProcess"}
            if missing and current != own_pid:
                # Windows retains the PID of an outer parent after that parent
                # exits. Known contiguous ancestors (including our Python
                # redirector) are already excluded; a nonexistent outer node
                # cannot itself be a currently matching fetch.
                return excluded
            raise FetchControlError(
                f"Controller ancestry verification failed: depth={depth}, pid={current}, "
                f"own_pid={own_pid}, error_type={type(exc).__name__}; no action taken.") from None
    raise FetchControlError("Controller ancestry exceeds the verified bound; no action taken.")


def stop_exact_fetch(root: Path, config: dict) -> dict:
    """Stop only a verified logical fetch, then rename unmarked pending files.

    No files or audit records are created unless the matched worker and all
    matching redirector parents have exited. Unsupported, inaccessible,
    ambiguous or changed identities fail closed. Nothing is deleted.
    """
    from .mbo_sir_data import build_requests, cache_directory

    root = _resolve_workspace(root)
    data_dir, out_dir = (_absolute(root, config[key]) for key in ("data_dir", "out_dir"))
    requests = build_requests(config)
    directories = []
    seen = set()
    for request in requests:
        directory = _inside(cache_directory(data_dir, request), data_dir)
        _inside(directory, root)
        if directory in seen:
            raise FetchControlError("Duplicate cache directory in frozen request grid; no action taken.")
        seen.add(directory)
        directories.append((directory, request))
    api = _get_process_api()
    excluded = _own_ancestry(api)
    matches = []
    inaccessible = False
    for process in api.process_iter():
        if int(process.pid) in excluded:
            continue
        try:
            if _matches(process, root):
                matches.append(process)
        except ProcessLookupError:
            continue
        except (PermissionError, OSError):
            inaccessible = True
        except Exception as exc:
            # psutil has its own NoSuchProcess/AccessDenied classes. Class
            # names are inspected, never exception text or other command lines.
            if type(exc).__name__ in {"NoSuchProcess", "ZombieProcess"}:
                continue
            if type(exc).__name__ == "AccessDenied":
                inaccessible = True
            else:
                raise FetchControlError("Process identity verification failed; no action taken.") from None
    if inaccessible:
        raise FetchControlError("One or more Python process identities are inaccessible; no action taken.")
    chain = _chain(matches)
    # Recheck every member immediately before termination, guarding PID reuse
    # and identity changes between enumeration and action.
    for process, created, parent in chain:
        try:
            if float(process.create_time()) != created or int(process.ppid()) != parent or not _matches(process, root):
                raise FetchControlError("Fetch identity changed before stopping; no action taken.")
        except FetchControlError:
            raise
        except Exception:
            raise FetchControlError("Fetch identity became unavailable before stopping; no action taken.") from None
    worker = chain[-1][0]
    deadline = time.monotonic() + _STOP_SECONDS
    try:
        worker.terminate()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        worker.wait(timeout=remaining)
        # Matching redirector parents must exit naturally. Never terminate a
        # parent or any process outside the verified matching chain.
        for parent, _, _ in reversed(chain[:-1]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            parent.wait(timeout=remaining)
    except Exception:
        raise FetchControlError("Fetch exit was not verified within twenty seconds; cache files remain untouched.") from None

    # All filesystem mutation starts below this point, after verified exit.
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "fetch_control.json"
    report = {
        "status": "stopped_preserving_prefixes", "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Transport-only resume with changed runtime concurrency; frozen research requests are unchanged.",
        "workspace": str(root), "worker_pid": int(worker.pid),
        "matched_chain": [{"pid": int(process.pid), "parent_pid": parent,
                           "started_utc": datetime.fromtimestamp(created, timezone.utc).isoformat()}
                          for process, created, parent in chain],
        "terminated_pids": [int(worker.pid)], "matching_parents_exited_naturally": len(chain) - 1,
        "completed_marker_directories_skipped": 0, "preserved_prefixes": [], "preserved_prefix_bytes": 0,
        "completed_receipts_modified": False, "deleted_files": 0,
    }
    _atomic_json(audit_path, report)
    try:
        for original_directory, request in directories:
            # Resolve again after the producer exited, guarding late symlink or
            # junction changes. Every target must remain in its exact cache.
            directory = _inside(original_directory, data_dir)
            if not directory.is_dir():
                continue
            marker = directory / "stream_complete.json"
            if marker.exists():
                _inside(marker, directory)
                content = json.loads(marker.read_text(encoding="utf-8"))
                if content.get("request") != request:
                    raise FetchControlError("A completed stream marker has different request provenance.")
                report["completed_marker_directories_skipped"] += 1
                continue
            pending = sorted(directory.glob("stream.*.dbn.zst"))
            if not pending:
                continue
            provenance = directory / "request.json"
            _inside(provenance, directory)
            if not provenance.is_file() or json.loads(provenance.read_text(encoding="utf-8")) != request:
                raise FetchControlError("Pending stream lacks matching exact-request provenance; all such files preserved in place.")
            for source in pending:
                if not source.is_file():
                    raise FetchControlError("A pending stream path is not a regular file.")
                _inside(source, directory)
                _inside(source, root)
                size = source.stat().st_size
                target = _inside(directory / f"interrupted.{uuid4().hex}.dbn.zst", directory)
                for attempt in range(12):
                    try:
                        source.rename(target)  # Never overwrite or delete.
                        break
                    except PermissionError:
                        if attempt == 11:
                            raise
                        time.sleep(min(.05 * 2 ** attempt, .5))
                report["preserved_prefixes"].append({"source": str(source.relative_to(root)),
                                                      "preserved_as": str(target.relative_to(root)), "bytes": size})
                report["preserved_prefix_bytes"] += size
                _atomic_json(audit_path, report)
        report["status"] = "stopped_prefixes_preserved"
        _atomic_json(audit_path, report)
        return report
    except Exception as exc:
        report["status"] = "stopped_prefix_preservation_failed"
        report["error_type"] = type(exc).__name__
        _atomic_json(audit_path, report)
        raise FetchControlError("Fetch stopped; some prefixes remain in place. Review the local transport audit before resuming.") from None


class _WindowsProcessAPI:
    """Minimal stdlib backend when psutil is absent (64-bit Windows only).

    Remote reads are limited to documented process-basic information and the
    fixed PEB process-parameter pointers for CurrentDirectory and CommandLine.
    No environment block or unrelated process memory is read or reported.
    """

    def __init__(self):
        if os.name != "nt":
            raise FetchControlError("psutil is unavailable and the native process backend requires Windows.")
        import ctypes as ct
        from ctypes import wintypes as wt
        if ct.sizeof(ct.c_void_p) != 8:
            raise FetchControlError("Native process verification requires 64-bit Python.")
        self.ct, self.wt = ct, wt
        self.kernel = ct.WinDLL("kernel32", use_last_error=True)
        self.ntdll = ct.WinDLL("ntdll", use_last_error=True)
        self.shell = ct.WinDLL("shell32", use_last_error=True)
        k = self.kernel
        k.OpenProcess.argtypes, k.OpenProcess.restype = [wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE
        k.CloseHandle.argtypes, k.CloseHandle.restype = [wt.HANDLE], wt.BOOL
        k.ReadProcessMemory.argtypes = [wt.HANDLE, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_size_t)]
        k.ReadProcessMemory.restype = wt.BOOL
        k.GetProcessTimes.argtypes = [wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4
        k.GetProcessTimes.restype = wt.BOOL
        k.IsWow64Process.argtypes, k.IsWow64Process.restype = [wt.HANDLE, ct.POINTER(wt.BOOL)], wt.BOOL
        k.TerminateProcess.argtypes, k.TerminateProcess.restype = [wt.HANDLE, wt.UINT], wt.BOOL
        k.WaitForSingleObject.argtypes, k.WaitForSingleObject.restype = [wt.HANDLE, wt.DWORD], wt.DWORD
        k.CreateToolhelp32Snapshot.argtypes, k.CreateToolhelp32Snapshot.restype = [wt.DWORD, wt.DWORD], wt.HANDLE
        self.ntdll.NtQueryInformationProcess.argtypes = [wt.HANDLE, wt.ULONG, ct.c_void_p, wt.ULONG, ct.POINTER(wt.ULONG)]
        self.ntdll.NtQueryInformationProcess.restype = wt.LONG
        self.shell.CommandLineToArgvW.argtypes = [wt.LPCWSTR, ct.POINTER(ct.c_int)]
        self.shell.CommandLineToArgvW.restype = ct.POINTER(wt.LPWSTR)
        k.LocalFree.argtypes, k.LocalFree.restype = [wt.HLOCAL], wt.HLOCAL

        class Entry(ct.Structure):
            _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("pid", wt.DWORD),
                        ("heap", ct.c_size_t), ("module", wt.DWORD), ("threads", wt.DWORD),
                        ("parent", wt.DWORD), ("priority", wt.LONG), ("flags", wt.DWORD),
                        ("name", wt.WCHAR * 260)]
        self.Entry = Entry
        k.Process32FirstW.argtypes, k.Process32FirstW.restype = [wt.HANDLE, ct.POINTER(Entry)], wt.BOOL
        k.Process32NextW.argtypes, k.Process32NextW.restype = [wt.HANDLE, ct.POINTER(Entry)], wt.BOOL

        class Basic(ct.Structure):
            _fields_ = [("reserved1", ct.c_void_p), ("peb", ct.c_void_p),
                        ("reserved2", ct.c_void_p * 2), ("pid", ct.c_size_t), ("reserved3", ct.c_void_p)]
        class Unicode(ct.Structure):
            _fields_ = [("length", wt.USHORT), ("maximum", wt.USHORT), ("buffer", ct.c_void_p)]
        self.Basic, self.Unicode = Basic, Unicode
        self._entries = None

    def _error(self):
        code = self.ct.get_last_error()
        if code in {87, 1168}:
            raise ProcessLookupError("Process is no longer available.")
        if code == 5:
            raise PermissionError("Process identity access denied.")
        raise OSError(code, "Native process identity operation failed.")

    @contextmanager
    def handle(self, pid: int, rights: int):
        handle = self.kernel.OpenProcess(rights, False, pid)
        if not handle:
            self._error()
        try:
            yield handle
        finally:
            self.kernel.CloseHandle(handle)

    def read(self, handle, address: int, size: int) -> bytes:
        # This method is called only for the fixed fields/bounded Unicode
        # buffers below, never exposed as a generic process-memory interface.
        if not address or not 0 < size <= 65_534:
            raise FetchControlError("Invalid bounded process-identity field.")
        buffer, copied = self.ct.create_string_buffer(size), self.ct.c_size_t()
        if not self.kernel.ReadProcessMemory(handle, address, buffer, size, self.ct.byref(copied)) or copied.value != size:
            self._error()
        return buffer.raw

    def parameter(self, handle, offset: int) -> str:
        if offset not in {0x38, 0x70}:
            raise FetchControlError("Unsupported fixed process identity field.")
        wow = self.wt.BOOL()
        if not self.kernel.IsWow64Process(handle, self.ct.byref(wow)):
            self._error()
        if wow.value:
            raise FetchControlError("Cross-architecture process identity is unsupported; no action taken.")
        basic, length = self.Basic(), self.wt.ULONG()
        status = self.ntdll.NtQueryInformationProcess(handle, 0, self.ct.byref(basic), self.ct.sizeof(basic), self.ct.byref(length))
        if status != 0 or not basic.peb:
            raise PermissionError("Process basic identity could not be read.")
        parameters = int.from_bytes(self.read(handle, int(basic.peb) + 0x20, 8), "little")
        field = self.Unicode.from_buffer_copy(self.read(handle, parameters + offset, self.ct.sizeof(self.Unicode)))
        if field.length > field.maximum or field.length % 2 or field.length > 65_534:
            raise FetchControlError("Malformed bounded process identity string.")
        return self.read(handle, field.buffer, field.length).decode("utf-16-le") if field.length else ""

    def creation_raw(self, handle) -> int:
        fields = [self.wt.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(handle, *(self.ct.byref(value) for value in fields)):
            self._error()
        return (int(fields[0].dwHighDateTime) << 32) | int(fields[0].dwLowDateTime)

    def _snapshot_entries(self):
        if self._entries is not None:
            return self._entries
        snapshot = self.kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == self.ct.c_void_p(-1).value:
            self._error()
        try:
            entries = []
            entry = self.Entry()
            entry.dwSize = self.ct.sizeof(entry)
            present = self.kernel.Process32FirstW(snapshot, self.ct.byref(entry))
            while present:
                entries.append((int(entry.pid), int(entry.parent), str(entry.name)))
                present = self.kernel.Process32NextW(snapshot, self.ct.byref(entry))
            self._entries = entries
            return entries
        finally:
            self.kernel.CloseHandle(snapshot)

    def parent_pid(self, pid):
        parents = {process: parent for process, parent, _ in self._snapshot_entries()}
        if pid not in parents:
            raise ProcessLookupError("Controller ancestor is no longer available.")
        return parents[pid]

    def process_iter(self):
        for pid, parent, name in self._snapshot_entries():
            if _PYTHON_NAME.fullmatch(name):
                yield _WindowsProcess(self, pid, parent, name)


class _WindowsProcess:
    def __init__(self, api, pid: int, parent: int, name: str):
        self.api, self.pid, self.parent, self._name = api, pid, parent, name
        self._creation_raw = None

    def name(self):
        return self._name

    def ppid(self):
        return self.parent

    def cwd(self):
        with self.api.handle(self.pid, 0x0400 | 0x0010) as handle:
            return self.api.parameter(handle, 0x38)

    def cmdline(self):
        with self.api.handle(self.pid, 0x0400 | 0x0010) as handle:
            command = self.api.parameter(handle, 0x70)
        count = self.api.ct.c_int()
        pointer = self.api.shell.CommandLineToArgvW(command, self.api.ct.byref(count))
        if not pointer:
            self.api._error()
        try:
            return [pointer[index] for index in range(count.value)]
        finally:
            self.api.kernel.LocalFree(self.api.ct.cast(pointer, self.api.wt.HLOCAL))

    def create_time(self):
        with self.api.handle(self.pid, 0x1000) as handle:
            raw = self.api.creation_raw(handle)
        if self._creation_raw is not None and self._creation_raw != raw:
            raise FetchControlError("Process ID was reused; no action taken.")
        self._creation_raw = raw
        return raw / 10_000_000 - 11_644_473_600

    def terminate(self):
        with self.api.handle(self.pid, 0x0001 | 0x1000 | 0x00100000) as handle:
            if self._creation_raw is None or self.api.creation_raw(handle) != self._creation_raw:
                raise FetchControlError("Process ID was reused before termination; no action taken.")
            if not self.api.kernel.TerminateProcess(handle, 1):
                self.api._error()

    def wait(self, timeout):
        try:
            with self.api.handle(self.pid, 0x1000 | 0x00100000) as handle:
                if self._creation_raw is None or self.api.creation_raw(handle) != self._creation_raw:
                    raise FetchControlError("Process ID was reused while verifying exit.")
                result = self.api.kernel.WaitForSingleObject(handle, max(1, int(timeout * 1000)))
                if result == 258:
                    raise TimeoutError("Process has not exited.")
                if result != 0:
                    self.api._error()
        except ProcessLookupError:
            return

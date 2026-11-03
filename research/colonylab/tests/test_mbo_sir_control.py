"""Exercise transport control exclusively with fake process identities."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from colonylab import mbo_sir_control as control
from colonylab import mbo_sir_data as data


class FakeProcess:
    def __init__(self, root, pid=100, parent=50, *, args=None, cwd=None, created=1_700_000_000., name="python.exe"):
        self.pid, self.parent, self.created, self.process_name = pid, parent, created, name
        self.root, self.directory = root, cwd or root
        self.command = args or ["python.exe", "-u", "scripts/run_mbo_sir.py", "--fetch-only"]
        self.terminated = False
        self.waited = []
        self.termination_hook = None
        self.wait_error = None
        self.create_calls = 0
        self.reuse_on_recheck = False

    def name(self): return self.process_name
    def cwd(self): return str(self.directory)
    def cmdline(self): return self.command
    def ppid(self): return self.parent
    def create_time(self):
        self.create_calls += 1
        return self.created + int(self.reuse_on_recheck and self.create_calls > 1)
    def terminate(self):
        self.terminated = True
        if self.termination_hook:
            self.termination_hook()
    def wait(self, timeout):
        self.waited.append(timeout)
        if self.wait_error:
            raise self.wait_error


class FakeAPI:
    def __init__(self, processes, *, own_parents=None):
        self.processes = processes
        self.parents = {999: 998, 998: 0} if own_parents is None else own_parents
    def parent_pid(self, pid):
        if pid not in self.parents:
            raise ProcessLookupError("Outer process no longer exists.")
        return self.parents[pid]
    def process_iter(self): return iter(self.processes)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_mbo_sir.py").write_text("# controlled fake script")
    # The managed Windows sandbox denies GetFinalPathNameByHandle on newly
    # created pytest directories. Treat the explicitly created fixture root as
    # a fake verified workspace; production still requires strict resolution.
    verified_root = root.absolute()
    actual_resolver = control._resolve_workspace
    monkeypatch.setattr(control, "_resolve_workspace", lambda value: verified_root
                        if Path(value).absolute() == verified_root else actual_resolver(value))
    config = {"data_dir": "data/study", "out_dir": "out/study"}
    requests = [{"start": "2026-06-01T00:00:00Z", "dataset": "A", "schema": "mbo", "limit": 100}]
    monkeypatch.setattr(data, "build_requests", lambda _: requests)
    monkeypatch.setattr(data, "cache_directory", lambda data_dir, request: data_dir / request["dataset"])
    monkeypatch.setattr(control.os, "getpid", lambda: 999)
    monkeypatch.setattr(control.time, "sleep", lambda _: None)
    return root, config, requests, monkeypatch


def install(setup, processes, own_parents=None):
    root, config, requests, monkeypatch = setup
    monkeypatch.setattr(control, "_get_process_api", lambda: FakeAPI(processes, own_parents=own_parents))
    return root, config, requests


def pending(root, request, *, token="abc", contents=b"partial"):
    directory = root / "data/study" / request["dataset"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "request.json").write_text(json.dumps(request))
    path = directory / f"stream.{token}.dbn.zst"
    path.write_bytes(contents)
    return directory, path


def test_exact_worker_exits_before_prefix_or_audit_mutation(setup):
    root, config, requests = install(setup, [])
    process = FakeProcess(root)
    install(setup, [process])
    directory, source = pending(root, requests[0])
    process.termination_hook = lambda: (source.exists() or pytest.fail("prefix mutated before termination"))
    result = control.stop_exact_fetch(root, config)
    assert process.terminated and len(process.waited) == 1 and process.waited[0] <= 20
    assert not source.exists()
    preserved = list(directory.glob("interrupted.*.dbn.zst"))
    assert len(preserved) == 1 and preserved[0].read_bytes() == b"partial"
    assert result["preserved_prefix_bytes"] == 7 and result["deleted_files"] == 0
    saved = json.loads((root / "out/study/fetch_control.json").read_text())
    assert saved["worker_pid"] == 100 and "command" not in json.dumps(saved)


def test_completed_marker_directory_and_receipts_are_untouched(setup):
    root, config, requests = install(setup, [])
    process = FakeProcess(root)
    install(setup, [process])
    directory, source = pending(root, requests[0])
    marker = directory / "stream_complete.json"
    marker.write_text(json.dumps({"request": requests[0], "download_file": source.name}))
    receipt = directory / "fetch.json"
    receipt.write_text('{"complete": true}')
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    result = control.stop_exact_fetch(root, config)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    assert result["completed_marker_directories_skipped"] == 1 and result["preserved_prefix_bytes"] == 0


def test_unique_linear_redirector_chain_terminates_only_leaf(setup):
    root = setup[0]
    parent, leaf = FakeProcess(root, 100, 50), FakeProcess(root, 101, 100, created=1_700_000_001.)
    root, config, _ = install(setup, [leaf, parent])
    result = control.stop_exact_fetch(root, config)
    assert leaf.terminated and not parent.terminated
    assert leaf.waited and parent.waited
    assert result["terminated_pids"] == [101] and result["matching_parents_exited_naturally"] == 1


def test_own_matching_python_launcher_is_excluded_with_entire_ancestry(setup):
    root = setup[0]
    own = FakeProcess(root, 999, 998)
    launcher = FakeProcess(root, 998, 997)
    grandparent = FakeProcess(root, 997, 0)
    existing = FakeProcess(root, 100, 50)
    root, config, _ = install(setup, [own, launcher, grandparent, existing], own_parents={999: 998, 998: 997, 997: 0})
    result = control.stop_exact_fetch(root, config)
    assert existing.terminated and not any(p.terminated for p in (own, launcher, grandparent))
    assert result["worker_pid"] == 100


@pytest.mark.parametrize("mode", ["none", "unrelated", "branch", "reused", "time_reversed"])
def test_ambiguous_or_changed_identity_never_terminates_or_mutates(setup, mode):
    root = setup[0]
    processes = [FakeProcess(root)]
    if mode == "none": processes = []
    if mode == "unrelated": processes += [FakeProcess(root, 101, 50)]
    if mode == "branch": processes += [FakeProcess(root, 101, 100), FakeProcess(root, 102, 100)]
    if mode == "reused": processes[0].reuse_on_recheck = True
    if mode == "time_reversed": processes += [FakeProcess(root, 101, 100, created=1_600_000_000.)]
    root, config, requests = install(setup, processes)
    _, source = pending(root, requests[0])
    with pytest.raises(control.FetchControlError):
        control.stop_exact_fetch(root, config)
    assert source.exists() and not any(p.terminated for p in processes)
    assert not (root / "out/study").exists()


@pytest.mark.parametrize("command,cwd_wrong", [
    (["python.exe", "other.py", "--fetch-only"], False),
    (["python.exe", "scripts/run_mbo_sir.py", "--analyze-only"], False),
    (["python.exe", "-c", "scripts/run_mbo_sir.py", "--fetch-only"], False),
    (["python.exe", "-m", "colonylab", "scripts/run_mbo_sir.py", "--fetch-only"], False),
    (["python.exe", "scripts/run_mbo_sir.py", "--fetch-only"], True),
])
def test_other_python_work_is_never_selected(setup, command, cwd_wrong):
    root = setup[0]
    process = FakeProcess(root, args=command, cwd=root.parent if cwd_wrong else root)
    root, config, _ = install(setup, [process])
    with pytest.raises(control.FetchControlError, match="No exact"):
        control.stop_exact_fetch(root, config)
    assert not process.terminated and not (root / "out/study").exists()


def test_exit_timeout_keeps_cache_and_audit_untouched(setup):
    root = setup[0]
    process = FakeProcess(root)
    process.wait_error = TimeoutError("private detail must not be exposed")
    root, config, requests = install(setup, [process])
    _, source = pending(root, requests[0])
    with pytest.raises(control.FetchControlError, match="twenty seconds") as error:
        control.stop_exact_fetch(root, config)
    assert "private detail" not in str(error.value)
    assert process.terminated and source.exists() and not (root / "out/study").exists()


def test_parent_exit_timeout_never_terminates_parent_or_renames_prefixes(setup):
    root = setup[0]
    parent, leaf = FakeProcess(root, 100, 50), FakeProcess(root, 101, 100)
    parent.wait_error = TimeoutError()
    root, config, requests = install(setup, [parent, leaf])
    _, source = pending(root, requests[0])
    with pytest.raises(control.FetchControlError):
        control.stop_exact_fetch(root, config)
    assert leaf.terminated and not parent.terminated and source.exists()
    assert not (root / "out/study").exists()


def test_only_pending_files_in_exact_request_directories_are_renamed(setup):
    root, config, requests = install(setup, [FakeProcess(setup[0])])
    directory, source = pending(root, requests[0])
    (directory / "sample.dbn.zst").write_bytes(b"sample")
    (directory / "interrupted.previous.dbn.zst").write_bytes(b"old")
    unrelated = root / "data/study/unrelated"
    unrelated.mkdir()
    (unrelated / "stream.other.dbn.zst").write_bytes(b"unrelated")
    control.stop_exact_fetch(root, config)
    assert not source.exists()
    assert (directory / "sample.dbn.zst").read_bytes() == b"sample"
    assert (directory / "interrupted.previous.dbn.zst").read_bytes() == b"old"
    assert (unrelated / "stream.other.dbn.zst").read_bytes() == b"unrelated"


def test_configured_external_path_fails_before_process_termination(setup):
    process = FakeProcess(setup[0])
    root, config, _ = install(setup, [process])
    config["data_dir"] = str(root.parent / "external")
    with pytest.raises(control.FetchControlError, match="escapes"):
        control.stop_exact_fetch(root, config)
    assert not process.terminated and not (root / "out/study").exists()


def test_unknown_controller_ancestry_fails_before_any_action(setup):
    process = FakeProcess(setup[0])
    root, config, _ = install(setup, [process], own_parents={})
    with pytest.raises(control.FetchControlError, match="ancestry"):
        control.stop_exact_fetch(root, config)
    assert not process.terminated


def test_inaccessible_python_identity_fails_closed_without_exposing_commandline(setup):
    root = setup[0]
    process = FakeProcess(root)
    process.cwd = lambda: (_ for _ in ()).throw(PermissionError("secret command detail"))
    root, config, _ = install(setup, [process])
    with pytest.raises(control.FetchControlError, match="inaccessible") as error:
        control.stop_exact_fetch(root, config)
    assert "secret command detail" not in str(error.value) and not process.terminated


def test_computed_cache_path_escape_fails_before_process_termination(setup):
    process = FakeProcess(setup[0])
    root, config, _ = install(setup, [process])
    setup[3].setattr(data, "cache_directory", lambda data_dir, request: root.parent / "outside")
    with pytest.raises(control.FetchControlError, match="escapes"):
        control.stop_exact_fetch(root, config)
    assert not process.terminated and not (root / "out/study").exists()


def test_other_workspace_python_command_line_is_never_read(setup):
    root = setup[0]
    other = FakeProcess(root, cwd=root.parent)
    other.cmdline = lambda: (_ for _ in ()).throw(AssertionError("other command line was read"))
    existing = FakeProcess(root, pid=101)
    root, config, _ = install(setup, [other, existing])
    control.stop_exact_fetch(root, config)
    assert existing.terminated and not other.terminated


@pytest.mark.parametrize("cwd", ["", "relative/workspace"])
def test_empty_or_relative_cwd_cannot_be_inferred_as_controller_workspace(setup, cwd):
    process = FakeProcess(setup[0])
    process.cwd = lambda: cwd
    root, config, requests = install(setup, [process])
    _, source = pending(root, requests[0])
    with pytest.raises(control.FetchControlError):
        control.stop_exact_fetch(root, config)
    assert not process.terminated and source.exists()
    assert not (root / "out/study").exists()


def test_exited_outer_parent_ends_ancestry_after_own_python_launcher_is_excluded(setup):
    root = setup[0]
    own, launcher, existing = FakeProcess(root, 999, 998), FakeProcess(root, 998, 997), FakeProcess(root, 100, 50)
    root, config, _ = install(setup, [own, launcher, existing], own_parents={999: 998, 998: 997})
    result = control.stop_exact_fetch(root, config)
    assert result["worker_pid"] == 100 and existing.terminated
    assert not own.terminated and not launcher.terminated


def test_inaccessible_outer_ancestor_is_not_treated_as_exited(setup):
    process = FakeProcess(setup[0])
    root, config, _ = install(setup, [process])
    api = FakeAPI([process])
    def parent(pid):
        if pid == 999:
            return 998
        raise PermissionError("private argument must never appear")
    api.parent_pid = parent
    setup[3].setattr(control, "_get_process_api", lambda: api)
    with pytest.raises(control.FetchControlError, match="error_type=PermissionError") as error:
        control.stop_exact_fetch(root, config)
    assert "depth=1" in str(error.value) and "pid=998" in str(error.value)
    assert "private argument" not in str(error.value) and not process.terminated

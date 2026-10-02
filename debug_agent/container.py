"""Docker container wrapper for the debug agent.

Two responsibilities: (1) uniform docker-exec interface for bash/pytest/pdb;
(2) transactional file edits — every write_file goes through a snapshot
stack so probes can always be reverted, even on exception.

terminate() only docker-stops containers launched via launch() (owned);
attach()-ed containers are left running."""
from __future__ import annotations

import base64
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return self.stdout + ("\n--- STDERR ---\n" + self.stderr if self.stderr else "")


class Container:
    def __init__(self, container_id: str, workdir: str):
        self.container_id = container_id
        self.workdir = workdir
        self._snapshots: dict[str, str] = {}
        self._owned = False

    @classmethod
    def launch(cls, image: str, workdir: str = "/work") -> "Container":
        name = f"dbg_{uuid.uuid4().hex[:10]}"
        # --entrypoint sleep overrides any ENTRYPOINT baked into the image
        # (e.g. SWE-bench Pro images set ENTRYPOINT=/bin/bash, which would
        # otherwise try to exec `bash sleep infinity` and fail).
        # --cap-add SYS_PTRACE + seccomp=unconfined: Delve (dlv) attaches to the
        # test process via ptrace, which the default container profile blocks.
        # Harmless for non-Go images, so applied unconditionally.
        subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", name,
             "-w", workdir, "--entrypoint", "sleep",
             "--cap-add", "SYS_PTRACE", "--security-opt", "seccomp=unconfined",
             image, "infinity"],
            check=True, capture_output=True,
        )
        c = cls(name, workdir)
        c._owned = True
        c.exec_bash(f"mkdir -p {shlex.quote(workdir)}")
        return c

    @classmethod
    def attach(cls, container_id: str, workdir: str) -> "Container":
        c = cls(container_id, workdir)
        c._owned = False
        return c

    def exec_bash(self, cmd: str, timeout: int = 120, stdin_data: str | None = None) -> ExecResult:
        proc = subprocess.run(
            ["docker", "exec", "-i", self.container_id, "bash", "-lc", cmd],
            input=stdin_data,
            capture_output=True, text=True, timeout=timeout, errors="replace",
        )
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)

    def ensure_dlv(self) -> None:
        """Make sure the Delve debugger is available at /go/bin/dlv (the path
        GoDlvSession invokes). Most Go SWE-bench images don't ship dlv. Prefer
        `docker cp` of the precompiled binary in assets/ (seconds) over a
        per-instance `go install` (network fetch + compile, minutes); fall back
        to the install if the asset is missing or won't run in this image.
        GOTOOLCHAIN=local pins the image's go (no network toolchain download);
        GOFLAGS= clears any -mod=readonly that would block the install."""
        r = self.exec_bash("test -x /go/bin/dlv && echo Y || echo N")
        if (r.stdout or "").strip().endswith("Y"):
            return
        if self._copy_precompiled_dlv():
            return
        self.exec_bash(
            "export PATH=/usr/local/go/bin:/go/bin:$PATH && "
            "GOTOOLCHAIN=local GOFLAGS= "
            "go install github.com/go-delve/delve/cmd/dlv@v1.25.1",
            timeout=600,
        )

    def _copy_precompiled_dlv(self) -> bool:
        """Copy the bundled dlv binary to /go/bin/dlv and confirm it runs.
        Returns False (so ensure_dlv falls back to `go install`) when the asset
        is absent, the copy fails, or the binary is incompatible with the image
        (verified via `dlv version`)."""
        asset = Path(__file__).resolve().parent / "assets" / "dlv-linux-amd64-go1.24"
        if not asset.exists():
            return False
        self.exec_bash("mkdir -p /go/bin")
        cp = subprocess.run(
            ["docker", "cp", str(asset), f"{self.container_id}:/go/bin/dlv"],
            capture_output=True, text=True,
        )
        if cp.returncode != 0:
            return False
        self.exec_bash("chmod +x /go/bin/dlv")
        v = self.exec_bash("/go/bin/dlv version 2>&1 | head -1")
        return "Delve" in (v.stdout or "")

    def read_file(self, path: str) -> str:
        r = self.exec_bash(f"cat {shlex.quote(path)}")
        return r.stdout

    def write_file(self, path: str, content: str) -> None:
        b64 = base64.b64encode(content.encode()).decode()
        q = shlex.quote(path)
        # Pipe the (possibly large) base64 via stdin rather than inlining it in
        # the command string: a big file would otherwise push the `bash -lc`
        # argument past the single-arg limit (MAX_ARG_STRLEN, 128 KiB) and raise
        # OSError(7, 'Argument list too long'), aborting RCA for that instance.
        self.exec_bash(
            f"mkdir -p $(dirname {q}) && base64 -d > {q}", stdin_data=b64
        )

    def snapshot(self, path: str) -> None:
        if path not in self._snapshots:
            self._snapshots[path] = self.read_file(path)

    def restore(self, path: str) -> None:
        if path in self._snapshots:
            self.write_file(path, self._snapshots[path])

    def restore_all(self) -> None:
        for p in list(self._snapshots):
            self.restore(p)

    def terminate(self) -> None:
        try:
            self.restore_all()
        finally:
            if self._owned:
                subprocess.run(
                    ["docker", "stop", self.container_id],
                    capture_output=True,
                )

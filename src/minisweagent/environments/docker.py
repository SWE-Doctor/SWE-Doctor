import logging
import os
import platform
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = ["--rm"]
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to use to execute commands. Default is ["bash", "-lc"].
    The actual command will be appended as argument to this. Override this to e.g., modify shell flags
    (e.g., to remove the `-l` flag to disable login shell) or to use python instead of bash to interpret commands.
    """


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        last_error: Exception | None = None
        for attempt in range(2):
            container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
            try:
                self.container_id = self._start_container_once(container_name)
                self._assert_container_running()
                return
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Container startup failed (attempt %d/2) for image %s: %s",
                    attempt + 1,
                    self.config.image,
                    exc,
                )
                # Best-effort cleanup before retrying.
                self.cleanup()
                self.container_id = None
                if attempt == 0:
                    self._force_pull_image()
        raise RuntimeError(
            f"Failed to start container for image {self.config.image} after pull retry: {last_error}"
        )

    def _start_container_once(self, container_name: str) -> str:
        """Run docker start command once and return container ID."""
        has_entrypoint_override = any(
            arg == "--entrypoint" or str(arg).startswith("--entrypoint=") for arg in self.config.run_args
        )
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
        ]
        # Many SWE-bench Pro images use ENTRYPOINT=/bin/bash. Without overriding entrypoint,
        # `docker run <image> sleep 2h` gets interpreted by bash as "run script named sleep",
        # causing immediate exit and later "No such container" failures on `docker exec`.
        if not has_entrypoint_override:
            cmd.extend(["--entrypoint", "/bin/sleep", self.config.image, self.config.container_timeout])
        else:
            cmd.extend([self.config.image, "sleep", self.config.container_timeout])
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        return result.stdout.strip()

    def _force_pull_image(self):
        """Force-refresh image from registry after startup/runtime failures."""
        self.logger.info("Pulling image again for recovery: %s", self.config.image)
        pull = subprocess.run(
            [self.config.executable, "pull", self.config.image],
            capture_output=True,
            text=True,
            timeout=max(self.config.pull_timeout, 300),
        )
        if pull.returncode != 0:
            msg = (pull.stderr or pull.stdout or "").strip()[:400]
            raise RuntimeError(f"Image re-pull failed for {self.config.image}: {msg}")

    def _assert_container_running(self):
        """Validate that the just-started container is actually running."""
        assert self.container_id, "Container not started"
        inspect_cmd = [
            self.config.executable,
            "inspect",
            "--format",
            "{{.State.Running}} {{.State.Status}} {{.State.ExitCode}}",
            self.container_id,
        ]
        inspect = subprocess.run(inspect_cmd, capture_output=True, text=True, timeout=15)
        if inspect.returncode != 0:
            msg = (inspect.stderr or inspect.stdout or "").strip()
            raise RuntimeError(
                f"Container {self.container_id} is unavailable immediately after startup: {msg}"
            )
        state = (inspect.stdout or "").strip().split()
        running = state[0].lower() == "true" if state else False
        if running:
            return
        logs = subprocess.run(
            [self.config.executable, "logs", self.container_id],
            capture_output=True,
            text=True,
            timeout=15,
        )
        log_excerpt = (logs.stderr or logs.stdout or "").strip()[:400]
        raise RuntimeError(
            f"Container {self.container_id} exited right after startup. "
            f"state='{(inspect.stdout or '').strip()}', logs='{log_excerpt}'"
        )

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if result.returncode != 0 and "No such container" in (result.stdout or ""):
                raise RuntimeError(
                    f"Container {self.container_id} disappeared during command execution."
                )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        if output["returncode"] != 0:
            return
        lines = output.get("output", "").splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
                submission = "".join(lines[i + 1:])
                raise Submitted(
                    {
                        "role": "exit",
                        "content": submission,
                        "extra": {"exit_status": "Submitted", "submission": submission},
                    }
                )

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()

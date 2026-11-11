"""Running a backend in another Python environment.

By default, backends run in the current Python environment. If a backend needs packages that conflict with
others (e.g. a different flair version), the model section can name a ``launcher``:

* ``requirements`` (easiest): SEFLAG starts the backend with ``uv run --with ...``. uv creates a separate,
  cached environment with the seflag package and these packages on first use; the current environment is not
  changed. Requires uv (https://docs.astral.sh/uv/).
* ``command``: any command that starts a Python interpreter with the seflag package installed, e.g. a conda env
  you created yourself: ``[conda, run, --no-capture-output, -n, seflag-flair, python]``.

SEFLAG appends ``-m seflag.worker <request> <output>`` to the command (see :mod:`seflag.worker`). The request
and output live in a temporary directory, and results record the command and the launched environment's
package versions.

Other tools work the same way, as long as the command ends in a Python interpreter with the seflag package
installed.
Docker, for example, can in principle run a backend with non-Python dependencies such as Morpheus, from an
image you build yourself with the seflag package and Morpheus inside. The temporary directory must be mounted at the same
path (shown for Linux, where it is under /tmp), and ``path`` refers to the location inside the container::

    model:
      backend: morpheus
      path: /morpheus
      launcher:
        command: [docker, run, --rm, -v, /tmp:/tmp, my-seflag-morpheus-image, python]

SEFLAG does not build or manage such images.
"""

from __future__ import annotations

import json
import shutil
import sys
from importlib.metadata import PackageNotFoundError, distribution

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seflag.errors import MissingDependencyError


class LauncherSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirements: list[str] | None = Field(
        None, description="Run the backend via uv in a separate environment with the seflag package and these packages.")
    python: str | None = Field(
        None, description="Python version for the uv environment, e.g. '3.12' (default: that of the current Python).")
    command: list[str] | None = Field(
        None, description="Command that starts a Python interpreter with the seflag package installed, e.g. conda run ... python.")

    @model_validator(mode="after")
    def _one_mode(self) -> LauncherSpec:
        if self.command is not None and (self.requirements is not None or self.python is not None):
            raise ValueError("use either 'command' or 'requirements'/'python', not both")
        if self.command is None and self.requirements is None:
            raise ValueError("give 'requirements' (run via uv) or 'command'")
        if self.command is not None and not self.command:
            raise ValueError("'command' must not be empty")
        return self


def seflag_requirement() -> tuple[str, bool]:
    """How the running seflag package can be installed elsewhere: (requirement or local path, editable)."""
    from seflag import __version__

    try:
        direct_url = distribution("seflag").read_text("direct_url.json")
    except PackageNotFoundError:
        direct_url = None
    if direct_url:
        info: dict = json.loads(direct_url)
        url: str = info.get("url", "")
        if url.startswith("file://") and "dir_info" in info:  # installed from a local source tree
            return url.removeprefix("file://"), bool(info["dir_info"].get("editable"))
    return f"seflag=={__version__}", False


def python_command(spec: LauncherSpec) -> list[str]:
    """The command prefix that starts the worker's Python interpreter."""
    if spec.command is not None:
        return list(spec.command)
    uv: str | None = shutil.which("uv")
    if uv is None:
        raise MissingDependencyError(
            "'launcher.requirements' runs the backend with uv, which is not installed. Install it "
            "(https://docs.astral.sh/uv/getting-started/installation/) or use 'launcher.command' instead.")
    seflag_req, editable = seflag_requirement()
    # same Python version as the current process unless requested otherwise: uv would otherwise pick the newest
    # one available, for which older packages (e.g. spaCy 3.7 models) may have no wheels
    python: str = spec.python or f"{sys.version_info.major}.{sys.version_info.minor}"
    command: list[str] = [uv, "run", "--no-project", "--isolated", "--python", python]
    command += ["--with-editable" if editable else "--with", seflag_req]
    for requirement in spec.requirements or []:
        command += ["--with", requirement]
    return command + ["python"]

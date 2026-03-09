"""
Run insurance-ebm tests on Databricks serverless compute.

Usage:
    uv run python run_tests_databricks.py
"""

import base64
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Load credentials
# ---------------------------------------------------------------------------

env_path = Path.home() / ".config/burning-cost/databricks.env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute as _compute
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

PROJECT_ROOT = Path(__file__).parent
WORKSPACE_PATH = "/Workspace/insurance-ebm"

# ---------------------------------------------------------------------------
# Upload project files
# ---------------------------------------------------------------------------

def upload_file(local_path: Path, remote_path: str) -> None:
    remote_dir = "/".join(remote_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=remote_dir)
    except Exception:
        pass
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        format=ImportFormat.AUTO,
        overwrite=True,
    )


SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache"}

print("Uploading project files to Databricks workspace...")
for fpath in sorted(PROJECT_ROOT.rglob("*")):
    if not fpath.is_file():
        continue
    if fpath.suffix not in (".py", ".toml", ".md", ".txt"):
        continue
    rel = fpath.relative_to(PROJECT_ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        continue
    remote = f"{WORKSPACE_PATH}/{rel}".replace("\\", "/")
    upload_file(fpath, remote)
    print(f"  Uploaded: {rel}")

print("Upload complete.")

# ---------------------------------------------------------------------------
# Create test runner notebook
# ---------------------------------------------------------------------------

# The notebook uses dbutils.notebook.exit() to return results so they appear
# in notebook_output.result, which the SDK can retrieve after the job finishes.

NOTEBOOK_CONTENT = """\
# Databricks notebook source
# MAGIC %pip install pytest polars matplotlib scikit-learn numpy openpyxl statsmodels --quiet

# COMMAND ----------

import subprocess, sys, os

# Install the package so its imports resolve correctly
r_install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-ebm", "--quiet", "--no-deps"],
    capture_output=True, text=True,
)

# Copy project to /tmp to avoid __pycache__ issues on Workspace FS
import shutil
shutil.copytree("/Workspace/insurance-ebm", "/tmp/insurance-ebm", dirs_exist_ok=True)

# Collect tests to verify discovery
r_collect = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/tmp/insurance-ebm/tests",
     "--collect-only", "-q",
     "--rootdir=/tmp/insurance-ebm",
    ],
    capture_output=True, text=True,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
         "PYTHONPATH": "/tmp/insurance-ebm/src"},
)
collect_out = r_collect.stdout[-1000:] + (r_collect.stderr[-300:] if r_collect.stderr else "")

# Run the actual tests from /tmp
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/tmp/insurance-ebm/tests",
     "-v", "--tb=short",
     "--rootdir=/tmp/insurance-ebm",
    ],
    capture_output=True, text=True,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
         "PYTHONPATH": "/tmp/insurance-ebm/src"},
)
test_out = result.stdout + ("\\nSTDERR:\\n" + result.stderr if result.stderr else "")

summary = (
    f"=== INSTALL (rc={r_install.returncode}) ===\\n{r_install.stderr[-200:] if r_install.stderr else 'ok'}\\n\\n"
    f"=== COLLECT (rc={r_collect.returncode}) ===\\n{collect_out}\\n\\n"
    f"=== TESTS (exit={result.returncode}) ===\\n"
    + (test_out[-6000:] if len(test_out) > 6000 else test_out)
)

# Return via dbutils so it appears in notebook_output.result
dbutils.notebook.exit(summary)
"""

test_nb_path = f"{WORKSPACE_PATH}/run_tests"
encoded_nb = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=test_nb_path,
    content=encoded_nb,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Test notebook uploaded to {test_nb_path}")

# ---------------------------------------------------------------------------
# Submit job with serverless compute
# ---------------------------------------------------------------------------

print("Submitting test job (serverless)...")
run_resp = w.jobs.submit(
    run_name="insurance-ebm-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            notebook_task=jobs.NotebookTask(
                notebook_path=test_nb_path,
                base_parameters={},
            ),
            environment_key="serverless",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="serverless",
            spec=_compute.Environment(client="2"),
        )
    ],
)

run_id = run_resp.run_id
print(f"Job run ID: {run_id}")

# ---------------------------------------------------------------------------
# Poll and collect results
# ---------------------------------------------------------------------------

print("Waiting for job to complete...")
while True:
    state = w.jobs.get_run(run_id=run_id)
    life = state.state.life_cycle_state.value if state.state.life_cycle_state else "UNKNOWN"
    result_state = state.state.result_state.value if state.state.result_state else None
    print(f"  State: {life} / {result_state}")
    if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(20)

# Retrieve notebook output
nb_result = ""
try:
    tasks = w.jobs.get_run(run_id=run_id).tasks or []
    for task in tasks:
        try:
            out = w.jobs.get_run_output(run_id=task.run_id)
            if out.notebook_output and out.notebook_output.result:
                nb_result = out.notebook_output.result
                break
        except Exception as e:
            print(f"  Could not retrieve task output: {e}")
except Exception as e:
    print(f"  Could not retrieve run details: {e}")

print("\n" + "=" * 60)
print(nb_result[-10000:] if len(nb_result) > 10000 else nb_result)
print("=" * 60)

if result_state != "SUCCESS":
    print(f"\nJob FAILED at infrastructure level. State: {result_state}")
    sys.exit(1)

# Check test exit code embedded in summary
import re
m = re.search(r"exit=(\d+)", nb_result)
test_exit = int(m.group(1)) if m else -1

if test_exit == 0:
    print("\nTests PASSED on Databricks.")
    sys.exit(0)
else:
    print(f"\nTests FAILED (exit code {test_exit}).")
    sys.exit(1)

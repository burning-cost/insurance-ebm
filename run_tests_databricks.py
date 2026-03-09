"""
Run insurance-ebm tests on Databricks serverless compute.

Usage:
    python run_tests_databricks.py
"""

import os
import sys
import time
import base64
from pathlib import Path

# Load credentials
env_path = Path.home() / ".config/burning-cost/databricks.env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

PROJECT_ROOT = Path(__file__).parent
WORKSPACE_PATH = "/Workspace/insurance-ebm"


def upload_directory(local_dir: Path, remote_base: str) -> None:
    """Upload all .py and .toml files to the Databricks workspace."""
    for fpath in sorted(local_dir.rglob("*")):
        if fpath.is_file() and fpath.suffix in (".py", ".toml", ".md", ".txt"):
            rel = fpath.relative_to(local_dir)
            if any(p in str(rel) for p in [".venv", "__pycache__", ".git", ".pytest_cache"]):
                continue
            remote_path = f"{remote_base}/{rel}".replace("\\", "/")
            remote_dir = "/".join(remote_path.split("/")[:-1])
            try:
                w.workspace.mkdirs(path=remote_dir)
            except Exception:
                pass
            content = fpath.read_bytes()
            encoded = base64.b64encode(content).decode()
            w.workspace.import_(
                path=remote_path,
                content=encoded,
                format=ImportFormat.AUTO,
                overwrite=True,
            )
            print(f"  Uploaded: {rel}")


print("Uploading project files to Databricks workspace...")
upload_directory(PROJECT_ROOT, WORKSPACE_PATH)
print("Upload complete.")

# ---------------------------------------------------------------------------
# Create test notebook
# ---------------------------------------------------------------------------

NOTEBOOK_CONTENT = '''# Databricks notebook source
# MAGIC %pip install pytest pytest-cov polars matplotlib scikit-learn numpy openpyxl statsmodels --quiet

# COMMAND ----------

import subprocess, sys, os

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-ebm/tests",
     "-v", "--tb=short",
     "--import-mode=importlib",
     "--rootdir=/Workspace/insurance-ebm",
    ],
    capture_output=True,
    text=True,
    env={
        **os.environ,
        "PYTHONPATH": "/Workspace/insurance-ebm/src",
    }
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
assert result.returncode == 0, f"Tests FAILED (exit {result.returncode})"
print("\\nAll tests passed.")
'''

test_nb_path = f"{WORKSPACE_PATH}/run_tests"
encoded = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=test_nb_path,
    content=encoded,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Test notebook uploaded to {test_nb_path}")

# ---------------------------------------------------------------------------
# Run as a one-time job using serverless compute
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
            # Serverless compute — no cluster spec needed
            environment_key="serverless",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="serverless",
            spec=jobs.JobEnvironmentSpec(
                client="2",
            ),
        )
    ],
)

run_id = run_resp.run_id
print(f"Job run ID: {run_id}")

# Poll until done
print("Waiting for job to complete...")
while True:
    state = w.jobs.get_run(run_id=run_id)
    life = state.state.life_cycle_state.value if state.state.life_cycle_state else "UNKNOWN"
    result_state = state.state.result_state.value if state.state.result_state else None
    print(f"  State: {life} / {result_state}")
    if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(20)

if result_state == "SUCCESS":
    print("\nTests PASSED on Databricks.")
    try:
        tasks = w.jobs.get_run(run_id=run_id).tasks or []
        for task in tasks:
            try:
                output = w.jobs.get_run_output(run_id=task.run_id)
                if output.notebook_output and output.notebook_output.result:
                    print(output.notebook_output.result[-3000:])
            except Exception:
                pass
    except Exception:
        pass
    sys.exit(0)
else:
    print(f"\nTests FAILED. Result: {result_state}")
    try:
        tasks = w.jobs.get_run(run_id=run_id).tasks or []
        for task in tasks:
            try:
                output = w.jobs.get_run_output(run_id=task.run_id)
                if output.notebook_output and output.notebook_output.result:
                    print(output.notebook_output.result[-5000:])
                if output.error:
                    print("Error:", output.error)
                if output.error_trace:
                    print("Trace:", output.error_trace[-2000:])
            except Exception as e:
                print(f"Could not retrieve task output: {e}")
    except Exception as e:
        print(f"Could not retrieve run details: {e}")
    sys.exit(1)

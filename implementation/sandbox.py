# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Optional


class Sandbox:
  """Base interface for executing generated code."""

  def __init__(self, timeout_seconds: int = 60):
    self.timeout_seconds = timeout_seconds

  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: Any,
      timeout_seconds: Optional[int] = None,
  ) -> tuple[Any, bool]:
    """Returns `function_to_run(test_input)` and whether execution succeeded."""
    raise NotImplementedError(
        'Must provide a sandbox for executing untrusted code.')


class ExecSandbox(Sandbox):
  """Simple subprocess executor for local examples.

  This is a timeout-bounded subprocess, not a security boundary. Use a stronger
  sandbox for untrusted model-generated code outside local experimentation.
  """

  _RESULT_PREFIX = "__ERA_SANDBOX_RESULT__"

  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: Any,
      timeout_seconds: Optional[int] = None,
  ) -> tuple[Any, bool]:
    timeout = timeout_seconds or self.timeout_seconds

    runner = (
        "import json\n"
        "import traceback\n\n"
        + program
        + "\n\n"
        + textwrap.dedent(
            f"""
            def _json_default(value):
              if hasattr(value, "tolist"):
                return value.tolist()
              if hasattr(value, "item"):
                return value.item()
              return str(value)

            try:
              result = {function_to_run}({test_input!r})
              payload = {{"success": True, "result": result}}
            except Exception:
              payload = {{
                  "success": False,
                  "error": "EXECUTION_ERROR: " + traceback.format_exc(),
              }}

            print(
                "{self._RESULT_PREFIX}"
                + json.dumps(payload, default=_json_default)
            )
            """
        )
    )

    with tempfile.TemporaryDirectory() as temp_dir:
      script_path = os.path.join(temp_dir, "run_solution.py")
      with open(script_path, "w") as f:
        f.write(runner)

      try:
        result = subprocess.run(
            [sys.executable, script_path],
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=temp_dir,
            check=False,
        )
      except subprocess.TimeoutExpired:
        return f"TIMEOUT: exceeded {timeout} seconds", False

    for line in reversed(result.stdout.splitlines()):
      if line.startswith(self._RESULT_PREFIX):
        payload = json.loads(line[len(self._RESULT_PREFIX):])
        if payload.get("success"):
          return payload.get("result"), True
        return payload.get("error"), False

    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    return f"NO_RESULT: subprocess exited with {result.returncode}\n{output}", False

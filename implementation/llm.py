import os
import random
import re
import shlex
import subprocess
import tempfile
import time
from typing import Optional, Protocol, Sequence


class LLM(Protocol):
    def draw_sample(self, prompt: str) -> str:
        ...


def _code_generation_prompt(prompt: str) -> str:
    return f"""
You are an expert Data Scientist and Python programmer.
Your task is to write Python code to solve a machine learning problem.
Return ONLY the python code.

Do not inspect files, modify files, run shell commands, or include explanations.

--- BEGIN PROMPT ---
{prompt}
--- END PROMPT ---
"""


def _strip_code_fences(content: str) -> str:
    content = content.strip()
    content = re.sub(r"^```python\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"^```\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n```$", "", content, flags=re.MULTILINE)
    return content.strip()


class GeminiLLM:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-image"):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "GeminiLLM requires the google-genai package. "
                "Install it with `pip install google-genai`."
            ) from exc

        self.client = genai.Client(api_key=api_key, vertexai=False)
        self.model_name = model_name

    def draw_sample(self, prompt: str) -> str:
        full_prompt = _code_generation_prompt(prompt)
        max_retries = 5
        base_delay = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                )
                return _strip_code_fences(response.text)
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"  [!] Rate limited (429). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"Gemini API Error: {e}")
                    raise e


class CodexLLM:
    """LLM adapter that uses the local Codex CLI and ChatGPT subscription auth."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        codex_binary: str = "codex",
        timeout_seconds: int = 600,
        profile: Optional[str] = None,
        extra_args: Sequence[str] = (),
    ):
        self.model_name = model_name
        self.codex_binary = codex_binary
        self.timeout_seconds = timeout_seconds
        self.profile = profile
        self.extra_args = tuple(extra_args)

    def draw_sample(self, prompt: str) -> str:
        full_prompt = _code_generation_prompt(prompt)

        with tempfile.NamedTemporaryFile("r+", suffix=".txt") as output_file:
            command = [
                self.codex_binary,
                "--ask-for-approval",
                "never",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-last-message",
                output_file.name,
            ]
            if self.model_name:
                command.extend(["--model", self.model_name])
            if self.profile:
                command.extend(["--profile", self.profile])
            command.extend(self.extra_args)
            command.append("-")

            max_retries = 5
            base_delay = 5
            for attempt in range(max_retries):
                try:
                    result = subprocess.run(
                        command,
                        input=full_prompt,
                        text=True,
                        capture_output=True,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError(
                        f"Codex CLI timed out after {self.timeout_seconds} seconds."
                    ) from exc

                output_file.seek(0)
                content = output_file.read()
                output_file.seek(0)
                output_file.truncate()

                if result.returncode == 0 and content:
                    return _strip_code_fences(content)

                error_text = result.stderr or result.stdout or content
                is_rate_limited = "429" in error_text or "rate limit" in error_text.lower()
                if is_rate_limited and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"  [!] Rate limited. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                raise RuntimeError(
                    "Codex CLI failed with exit code "
                    f"{result.returncode}:\n{error_text.strip()}"
                )

        raise RuntimeError("Codex CLI did not produce a response.")


def create_llm_from_env() -> LLM:
    provider = os.environ.get("ERA_LLM_PROVIDER", "codex").strip().lower()

    if provider == "codex":
        extra_args = shlex.split(os.environ.get("CODEX_EXTRA_ARGS", ""))
        timeout = int(os.environ.get("CODEX_TIMEOUT_SECONDS", "600"))
        return CodexLLM(
            model_name=os.environ.get("CODEX_MODEL") or None,
            codex_binary=os.environ.get("CODEX_BINARY", "codex"),
            timeout_seconds=timeout,
            profile=os.environ.get("CODEX_PROFILE") or None,
            extra_args=extra_args,
        )

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use ERA_LLM_PROVIDER=gemini.")
        return GeminiLLM(
            api_key=api_key,
            model_name=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-image"),
        )

    raise ValueError(f"Unsupported ERA_LLM_PROVIDER: {provider!r}. Use 'codex' or 'gemini'.")

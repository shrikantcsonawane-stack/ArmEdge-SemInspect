"""Llama 3.2 1B Instruct advisor running on-device through ExecuTorch.

Preferred path: the official ExecuTorch Python LLM runner
(`executorch.extension.llm.runner.TextLLMRunner`, shipped in the `executorch` pip wheel).
Alternative: the official C++ runner binary `llama_main` (examples/models/llama).
Both need an exported Llama `.pte` + `tokenizer.model` (see models/README.md).
No cloud API is used. If anything is missing, `status()` reports
"ExecuTorch model not configured" and the caller uses the labelled fallback.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = ("You are a semiconductor wafer quality-control assistant. The vision system's result is the "
          "source of truth: never change the defect class, confidence, severity or decision. Using only "
          "the JSON provided, write four short labelled lines: Defect explanation, Severity explanation, "
          "Recommended operator action, Secondary inspection (yes/no and why).")


def build_prompt(facts: dict) -> str:
    """Llama 3 instruct chat template (BOS is added by the runner via export metadata)."""
    return ("<|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInspection result:\n" +
            json.dumps(facts, indent=1) +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def python_runner_available() -> bool:
    try:
        from executorch.extension.llm.runner import TextLLMRunner  # noqa: F401
        return True
    except Exception:
        return False


class ExecuTorchLlamaAdvisor:
    def __init__(self, cfg: dict):
        a = cfg["advisor"]
        self.runner_bin = ROOT / a["runner_binary"]
        self.pte = ROOT / a["model_pte"]
        self.tok = ROOT / a["tokenizer"]
        self.max_new = int(a.get("max_new_tokens", 160))
        self.timeout = int(a.get("timeout_s", 60))
        self._runner = None

    def mode(self) -> str | None:
        if python_runner_available():
            return "python TextLLMRunner"
        if self.runner_bin.exists():
            return "C++ llama_main"
        return None

    def status(self) -> dict:
        missing = [str(p.relative_to(ROOT)) for p in (self.pte, self.tok) if not p.exists()]
        if self.mode() is None:
            missing.append("ExecuTorch LLM runner (pip install executorch, or build llama_main)")
        return {"configured": not missing, "missing": missing, "runner": self.mode(),
                "message": "ExecuTorch Llama ready" if not missing else "ExecuTorch model not configured"}

    def generate(self, facts: dict) -> str:
        if not self.status()["configured"]:
            raise RuntimeError("ExecuTorch model not configured")
        prompt = build_prompt(facts)
        if self.mode() == "python TextLLMRunner":
            from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
            if self._runner is None:
                self._runner = TextLLMRunner(str(self.pte), str(self.tok))
            self._runner.reset()
            toks: list[str] = []
            cfg = GenerationConfig(echo=False, max_new_tokens=self.max_new, temperature=0.0)
            self._runner.generate(prompt, cfg, token_callback=toks.append)
            text = "".join(toks)
        else:
            cmd = [str(self.runner_bin), f"--model_path={self.pte}", f"--tokenizer_path={self.tok}",
                   f"--prompt={prompt}", "--temperature=0"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, check=True).stdout
            text = out.split("<|end_header_id|>")[-1]  # runner echoes the prompt
        return text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

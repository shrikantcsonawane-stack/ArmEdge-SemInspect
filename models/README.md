# Models (not committed)

| File | Produced by |
|---|---|
| `mobilenetv3_small_best.pt` | `scripts/train.py` (best checkpoint + class names + metadata) |
| `training_log.json` | `scripts/train.py` (actual measured run log) |
| `classifier.pte` | `python -m src.inference --export-pte` (ExecuTorch, XNNPACK delegate) |
| `llama/llama3_2_1b_instruct.pte`, `llama/tokenizer.model` | manual steps below |

## Llama 3.2 1B Instruct with ExecuTorch (pending: not yet performed)
Follows https://docs.pytorch.org/executorch/stable/llm/export-llm.html. Check the flags against your installed ExecuTorch version; this project used 1.6.0.

1. Accept Meta's licence and download the **original** checkpoint for `meta-llama/Llama-3.2-1B-Instruct`. You need `consolidated.00.pth`, `params.json` and `tokenizer.model`.
2. Export with 4-bit weights for XNNPACK (Arm CPU):
```bash
pip install executorch
CFG=$(python -c "import executorch,os;print(os.path.dirname(executorch.__file__))")/examples/models/llama/config/llama_xnnpack.yaml
python -m executorch.extension.llm.export.export_llm --config $CFG \
  +base.model_class=llama3_2 \
  +base.checkpoint=consolidated.00.pth +base.params=params.json \
  +export.max_seq_length=1024 +export.max_context_length=1024 \
  +export.output_name=models/llama/llama3_2_1b_instruct.pte
cp tokenizer.model models/llama/
```
3. Restart the app. **System status** should show "ExecuTorch Llama ready", and recommendations will then be labelled "ExecuTorch Llama 3.2 1B Instruct".

The runner is `executorch.extension.llm.runner.TextLLMRunner`, which ships in the pip wheel, so no C++ build is needed. You can also place the official C++ `llama_main` binary at `models/llama/llama_main`.

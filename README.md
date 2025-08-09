# Attention + CEM (CUDA-ready)

This repo now supports device-agnostic execution: it runs on CPU (for macOS without NVIDIA GPUs) and seamlessly uses NVIDIA GPUs when available.

## Quick Start

- CPU sanity test (no training):
  - `python test_full_pipeline.py`
  - It constructs the attention classifier and runs a forward pass on dummy features.

- NVIDIA GPU run (training/inference):
  1. Install the CUDA build of PyTorch that matches your driver:
     - CUDA 12.1: `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
     - CUDA 11.8: `pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio`
  2. Verify: `python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`
  3. Example: `bash run_exp.sh` (or run `main_MIA.py` with your flags).

## What changed

- Added unified device management in `model_training_attention.py` via `self.device`.
- Moved core modules (`f`, `f_tail`, `classifier`, `attention_classifier`, AE) to `self.device`.
- All temporary tensors and noises in the main train/val paths are created on the correct device.

These changes prevent CPU/GPU device mismatch errors and make the attention + CEM flow CUDA-ready.

## Notes for macOS (no NVIDIA GPU)

- You can develop and sanity-check on CPU with the test script above.
- If you hit a local PyTorch binary mismatch (arm64 vs x86_64), install the wheel that matches your Python architecture.

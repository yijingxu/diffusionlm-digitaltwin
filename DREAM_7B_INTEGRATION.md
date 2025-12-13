# Dream 7B Integration for Digital Twin Simulation

This document describes how to use **Dream 7B** (a diffusion language model) instead of OpenAI/Gemini API models for digital twin simulation.

## Overview

**Dream 7B** is an open-source diffusion large language model that achieves competitive performance with autoregressive models. Unlike API-based models, Dream runs locally on your GPU, which means:
- ✅ No API costs
- ✅ Full control over the model
- ✅ Privacy (data stays local)
- ⚠️ Requires significant GPU memory (20GB+)
- ⚠️ Slower than API models for batch processing

**Model Resources:**
- HuggingFace: https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
- GitHub: https://github.com/DreamLM/Dream
- Blog: https://hkunlp.github.io/blog/2025/dream/

## Key Differences from Original Implementation

### 1. **Model Architecture**
- **Original**: Autoregressive models (GPT-4.1-mini, Gemini) via API
- **Dream**: Diffusion language model running locally

### 2. **Generation Method**
- **Original**: Standard `generate()` with autoregressive sampling
- **Dream**: `diffusion_generate()` with diffusion timesteps and remasking strategies

### 3. **Context Length**
- **Original**: Up to 128K tokens (GPT-4.1-mini)
- **Dream**: 2048 tokens total (input + output)

### 4. **Concurrency**
- **Original**: 300+ concurrent API requests
- **Dream**: Typically 1 concurrent request (GPU memory limited)

## Installation

### Prerequisites
1. **GPU with at least 20GB memory** (e.g., A100, A6000, or similar)
2. **Python 3.11+**

### Required Packages

```bash
# Install specific versions required by Dream
pip install transformers==4.46.2
pip install torch==2.5.1

# Other dependencies (should already be installed)
pip install tqdm asyncio pyyaml
```

**Important**: Dream requires these specific versions. Other versions may not work correctly.

### Check Dependencies

Before running simulations, check if all dependencies are correctly installed:

```bash
python check_dream_dependencies.py
```

This script will verify:
- ✓ Python version (3.11+)
- ✓ PyTorch installation and CUDA availability
- ✓ Transformers version (4.46.2)
- ✓ Other required packages
- ✓ GPU memory (20GB+ recommended)
- ✓ Dream model access from HuggingFace

**Example output:**
```
Dream 7B Dependency Checker
============================================================

============================================================
  Python Version
============================================================
✓ Python 3.11.7

============================================================
  PyTorch & CUDA
============================================================
✓ PyTorch 2.5.1 with CUDA - NVIDIA A100 (40.0 GB)
  Device: NVIDIA A100-SXM4-40GB
  Total Memory: 40.0 GB
  CUDA Version: 12.1

============================================================
  Transformers
============================================================
✓ transformers 4.46.2 (required: 4.46.2)

✓ All critical dependencies are installed!
✓ Ready to run Dream 7B simulations!
```

## Files Created

### 1. `text_simulation/llm_helper_dream.py`
New LLM helper module for Dream 7B integration.

**Key Features:**
- Loads Dream model once (singleton pattern) and reuses it
- Handles prompt truncation for 2048 token limit
- Implements diffusion generation with configurable parameters
- Maintains same interface as original `llm_helper.py` for compatibility

**Key Classes:**
- `DreamLLMConfig`: Configuration class for Dream-specific parameters
- `process_prompts_batch_dream()`: Batch processing function

**Key Functions:**
- `_load_dream_model()`: Loads model and tokenizer (singleton)
- `_get_dream_response_direct()`: Generates response using diffusion
- `_truncate_prompt_if_needed()`: Handles context length limits

### 2. `text_simulation/run_LLM_simulations_dream.py`
Main simulation script adapted for Dream 7B.

**Usage:**
```bash
python text_simulation/run_LLM_simulations_dream.py \
    --config text_simulation/configs/dream_config.yaml \
    --max_personas 5
```

**Differences from original:**
- Uses `DreamLLMConfig` instead of `LLMConfig`
- Uses `process_prompts_batch_dream()` instead of `process_prompts_batch()`
- Outputs saved to `text_simulation_output_dream/` (separate from original)

### 3. `text_simulation/configs/dream_config.yaml`
Configuration file for Dream 7B simulation.

**Key Parameters:**
- `model_name`: "Dream-org/Dream-v0-Instruct-7B"
- `temperature`: 0.2 (lower for deterministic responses)
- `steps`: 512 (diffusion timesteps)
- `alg`: "entropy" (remasking strategy)
- `max_context_length`: 2048
- `num_workers`: 1 (local model, GPU memory limited)

## Configuration Parameters

### Dream-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 512 | Number of diffusion timesteps. More steps = better quality but slower |
| `alg` | "entropy" | Remasking strategy: `"origin"`, `"maskgit_plus"`, `"topk_margin"`, `"entropy"` |
| `alg_temp` | 0.0 | Randomness for confidence-based strategies |
| `max_context_length` | 2048 | Total context limit (input + output tokens) |

### Remasking Strategies (`alg`)

1. **`"origin"`**: Random token generation order (baseline)
2. **`"maskgit_plus"`**: Based on top-1 confidence
3. **`"topk_margin"`**: Based on margin confidence (top1 - top2)
4. **`"entropy"`**: Based on entropy of token distribution (recommended)

## Usage Workflow

### Step 1: Prepare Data (Same as Original)

```bash
# Download dataset
python download_dataset.py

# Convert personas to text
python text_simulation/batch_convert_personas.py \
    --persona_json_dir data/mega_persona_json/mega_persona \
    --output_text_dir text_simulation/text_personas \
    --variant full

# Convert questions to text
python text_simulation/convert_question_json_to_text.py

# Create simulation inputs
python text_simulation/create_text_simulation_input.py
```

### Step 2: Run Dream 7B Simulation

```bash
# Test with 5 personas
python text_simulation/run_LLM_simulations_dream.py \
    --config text_simulation/configs/dream_config.yaml \
    --max_personas 5

# Run all personas (will take much longer)
python text_simulation/run_LLM_simulations_dream.py \
    --config text_simulation/configs/dream_config.yaml
```

### Step 3: Evaluate Results (Same as Original)

```bash
# Update evaluation config to point to Dream output directory
# Edit evaluation/evaluation_basic.yaml:
#   trial_dir: "text_simulation/text_simulation_output_dream/"

# Run evaluation
./scripts/run_evaluation_pipeline.sh
```

## Handling Context Length Limits

Dream has a **2048 token context limit** (input + output). The implementation automatically:

1. **Truncates prompts** if they exceed the limit
2. **Reserves tokens** for output (default: 512 tokens)
3. **Keeps the end** of the prompt (where the question is)

**If prompts are too long:**
- Consider using `variant="summary"` or `variant="summary+text"` in persona conversion
- This uses persona summaries instead of full response history
- Reduces prompt length while maintaining key information

```bash
# Use summary variant for shorter prompts
python text_simulation/batch_convert_personas.py \
    --persona_json_dir data/mega_persona_json/mega_persona \
    --output_text_dir text_simulation/text_personas \
    --variant summary+text
```

## Performance Considerations

### Speed
- **Dream**: ~1-5 seconds per prompt (depending on GPU and steps)
- **API models**: ~0.5-2 seconds per prompt (depending on API latency)
- **Batch processing**: Dream is slower due to sequential processing (GPU memory limits concurrency)

### Memory
- **Dream**: Requires 20GB+ GPU memory
- **API models**: No local GPU memory needed

### Cost
- **Dream**: Free (runs locally)
- **API models**: Pay per token (can be expensive for large batches)

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
- Reduce `num_workers` to 1 (already default)
- Reduce `max_new_tokens` or `steps`
- Use a GPU with more memory
- Process fewer personas at a time

### Issue: "Prompt too long"
**Solution:**
- Use `variant="summary"` for persona conversion
- Reduce `max_new_tokens` to reserve more tokens for input
- Manually truncate very long persona profiles

### Issue: "Model loading fails"
**Solution:**
- Ensure `transformers==4.46.2` and `torch==2.5.1`
- Check GPU memory availability
- Try loading on CPU first (will be very slow): `device: "cpu"`

### Issue: "Generation quality is poor"
**Solution:**
- Increase `steps` (more diffusion timesteps = better quality)
- Try different `alg` values (`"entropy"` is recommended)
- Adjust `temperature` (lower = more deterministic, higher = more diverse)

## Comparison with Original Results

To compare Dream 7B results with GPT-4.1-mini:

1. **Run both models** on the same personas
2. **Use same evaluation pipeline** (point to different output directories)
3. **Compare MAD accuracy metrics** between the two

Expected differences:
- Dream may have slightly lower accuracy (smaller model, different architecture)
- Dream may be more consistent (deterministic with temperature=0.2)
- Dream may handle certain question types differently

## Future Improvements

Potential enhancements:
1. **Batch processing**: Process multiple prompts in parallel (if GPU memory allows)
2. **Prompt optimization**: Better truncation strategies for long prompts
3. **Fine-tuning**: Fine-tune Dream on digital twin data (if training code available)
4. **Quantization**: Use quantized model to reduce memory requirements

## References

- Dream 7B Paper: [arXiv:2508.15487](https://arxiv.org/abs/2508.15487)
- Dream Blog: https://hkunlp.github.io/blog/2025/dream/
- Dream GitHub: https://github.com/DreamLM/Dream
- HuggingFace Model: https://huggingface.co/Dream-org/Dream-v0-Instruct-7B

## Summary

This integration allows you to:
1. ✅ Run digital twin simulations using Dream 7B locally
2. ✅ Avoid API costs
3. ✅ Maintain privacy (data stays local)
4. ✅ Compare diffusion vs. autoregressive models

**Trade-offs:**
- ⚠️ Requires powerful GPU (20GB+ memory)
- ⚠️ Slower than API models
- ⚠️ Limited context length (2048 tokens)

The implementation maintains compatibility with the original evaluation pipeline, so you can directly compare results between Dream 7B and GPT-4.1-mini.


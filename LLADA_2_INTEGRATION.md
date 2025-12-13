# LLaDA 2.0 Integration for Digital Twin Simulation

This document describes how to use **LLaDA 2.0** (Large Language Diffusion Model) instead of OpenAI/Gemini API models or Dream for digital twin simulation.

## Overview

**LLaDA 2.0** is a large-scale masked diffusion language model (8B parameters) that demonstrates competitive performance with autoregressive models like LLaMA3 8B. Unlike API-based models, LLaDA runs locally on your GPU, which means:
- ✅ No API costs
- ✅ Full control over the model
- ✅ Privacy (data stays local)
- ✅ Novel diffusion-based generation approach
- ⚠️ Requires significant GPU memory (20GB+)
- ⚠️ Slower than API models for batch processing

**Model Resources:**
- Paper: [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- GitHub: https://github.com/ML-GSAI/LLaDA
- Demo: https://ml-gsai.github.io/LLaDA-demo/
- HuggingFace: Check the GitHub repo for model release

## Key Differences from Original Implementation

### 1. **Model Architecture**
- **Original**: Autoregressive models (GPT-4.1-mini, Gemini) via API
- **LLaDA**: Masked diffusion language model running locally

### 2. **Generation Method**
- **Original**: Standard `generate()` with autoregressive sampling (left-to-right)
- **LLaDA**: Masked diffusion generation with flexible remasking strategies

### 3. **Context Length**
- **Original**: Up to 128K tokens (GPT-4.1-mini)
- **LLaDA**: 2048 tokens total (input + output)

### 4. **Concurrency**
- **Original**: 300+ concurrent API requests
- **LLaDA**: Typically 1 concurrent request (GPU memory limited)

## Installation

### Prerequisites
1. **GPU with at least 20GB memory** (e.g., A100, A6000, or similar)
2. **Python 3.11+**

### Required Packages

```bash
# Install transformers and torch
# Check LLaDA GitHub repo for specific version requirements
pip install transformers torch

# Other dependencies (should already be installed)
pip install tqdm asyncio pyyaml
```

**Important**: Check the LLaDA GitHub repository for specific version requirements. The model may require specific versions of transformers and torch.

### Model Access

The LLaDA model should be available on HuggingFace. Check the GitHub repository for the exact model path:
- Expected path: `ML-GSAI/LLaDA-8B-Instruct` (update in config if different)

## Files Created

### 1. `text_simulation/llm_helper_llada.py`
New LLM helper module for LLaDA 2.0 integration.

**Key Features:**
- Loads LLaDA model once (singleton pattern) and reuses it
- Handles prompt truncation for 2048 token limit
- Implements masked diffusion generation with configurable parameters
- Maintains same interface as original `llm_helper.py` for compatibility

**Key Classes:**
- `LLaDALLMConfig`: Configuration class for LLaDA-specific parameters
- `process_prompts_batch_llada()`: Batch processing function

**Key Functions:**
- `_load_llada_model()`: Loads model and tokenizer (singleton)
- `_get_llada_response_direct()`: Generates response using masked diffusion
- `_truncate_prompt_if_needed()`: Handles context length limits

### 2. `text_simulation/run_LLM_simulations_llada.py`
Main simulation script adapted for LLaDA 2.0.

**Usage:**
```bash
python text_simulation/run_LLM_simulations_llada.py \
    --config text_simulation/configs/llada_config.yaml \
    --max_personas 5
```

**Differences from original:**
- Uses `LLaDALLMConfig` instead of `LLMConfig`
- Uses `process_prompts_batch_llada()` instead of `process_prompts_batch()`
- Outputs saved to `text_simulation_output_llada/` (separate from original)

### 3. `text_simulation/configs/llada_config.yaml`
Configuration file for LLaDA 2.0 simulation.

**Key Parameters:**
- `model_name`: "ML-GSAI/LLaDA-8B-Instruct" (update when model is released)
- `temperature`: 0.2 (lower for deterministic responses)
- `steps`: 512 (diffusion timesteps)
- `remasking_strategy`: "random" (remasking strategy for masked diffusion)
- `guidance_scale`: 1.0 (classifier-free guidance)
- `max_context_length`: 2048
- `num_workers`: 1 (local model, GPU memory limited)

## Configuration Parameters

### LLaDA-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 512 | Number of diffusion timesteps. More steps = better quality but slower |
| `remasking_strategy` | "random" | Remasking strategy: `"random"`, `"entropy"`, `"confidence"`, etc. |
| `guidance_scale` | 1.0 | Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger) |
| `max_context_length` | 2048 | Total context limit (input + output tokens) |

### Remasking Strategies

LLaDA uses masked diffusion with flexible remasking. The remasking strategy determines which tokens to mask/unmask at each diffusion step:

1. **`"random"`**: Random token generation order (baseline)
2. **`"entropy"`**: Based on entropy of token distribution
3. **`"confidence"`**: Based on model confidence scores
4. Other strategies may be available (check LLaDA documentation)

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

### Step 2: Run LLaDA 2.0 Simulation

```bash
# Test with 5 personas
python text_simulation/run_LLM_simulations_llada.py \
    --config text_simulation/configs/llada_config.yaml \
    --max_personas 5

# Run all personas (will take much longer)
python text_simulation/run_LLM_simulations_llada.py \
    --config text_simulation/configs/llada_config.yaml
```

### Step 3: Evaluate Results (Same as Original)

```bash
# Update evaluation config to point to LLaDA output directory
# Edit evaluation/evaluation_basic.yaml:
#   trial_dir: "text_simulation/text_simulation_output_llada/"

# Run evaluation
./scripts/run_evaluation_pipeline.sh
```

## Handling Context Length Limits

LLaDA has a **2048 token context limit** (input + output). The implementation automatically:

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
- **LLaDA**: ~1-10 seconds per prompt (depending on GPU, steps, and remasking strategy)
- **API models**: ~0.5-2 seconds per prompt (depending on API latency)
- **Batch processing**: LLaDA is slower due to sequential processing (GPU memory limits concurrency)

### Memory
- **LLaDA**: Requires 20GB+ GPU memory (8B model)
- **API models**: No local GPU memory needed

### Cost
- **LLaDA**: Free (runs locally)
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
- Check LLaDA GitHub repo for specific version requirements
- Ensure GPU memory availability
- Try loading on CPU first (will be very slow): `device: "cpu"`
- Verify model path is correct on HuggingFace

### Issue: "Generation quality is poor"
**Solution:**
- Increase `steps` (more diffusion timesteps = better quality)
- Try different `remasking_strategy` values
- Adjust `guidance_scale` (higher = stronger guidance)
- Adjust `temperature` (lower = more deterministic, higher = more diverse)

### Issue: "Model not found on HuggingFace"
**Solution:**
- Check LLaDA GitHub repository for model release status
- Update `model_name` in config with correct path
- The model may need to be downloaded from a different source

## Comparison with Other Models

### LLaDA vs. Dream
- **LLaDA**: 8B parameters, masked diffusion, competitive with LLaMA3 8B
- **Dream**: 7B parameters, diffusion model, different architecture
- Both use diffusion-based generation but with different approaches

### LLaDA vs. Autoregressive Models
- **LLaDA**: Non-autoregressive generation (can generate tokens in any order)
- **Autoregressive**: Left-to-right generation
- LLaDA may handle certain tasks differently due to its generation method

## Replicating LLaDA Results

To replicate the results from the LLaDA paper:

1. **Use the same model checkpoint** as reported in the paper
2. **Match the generation parameters** (steps, remasking strategy, etc.)
3. **Use the same evaluation metrics** as the paper
4. **Compare with baseline models** (LLaMA3 8B, etc.)

**Key parameters from the paper:**
- Model size: 8B parameters
- Training: Pretrained on 2.3T tokens, then SFT
- Generation: Masked diffusion with flexible remasking
- Performance: Competitive with LLaMA3 8B on various benchmarks

## Future Improvements

Potential enhancements:
1. **Batch processing**: Process multiple prompts in parallel (if GPU memory allows)
2. **Prompt optimization**: Better truncation strategies for long prompts
3. **Fine-tuning**: Fine-tune LLaDA on digital twin data (if training code available)
4. **Quantization**: Use quantized model to reduce memory requirements
5. **Optimization**: Use consistency distillation or other speed-up techniques

## References

- LLaDA Paper: [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- LLaDA GitHub: https://github.com/ML-GSAI/LLaDA
- LLaDA Demo: https://ml-gsai.github.io/LLaDA-demo/
- HuggingFace: https://huggingface.co/papers/2502.09992

## Summary

This integration allows you to:
1. ✅ Run digital twin simulations using LLaDA 2.0 locally
2. ✅ Avoid API costs
3. ✅ Maintain privacy (data stays local)
4. ✅ Compare diffusion vs. autoregressive models
5. ✅ Replicate results from the LLaDA paper

**Trade-offs:**
- ⚠️ Requires powerful GPU (20GB+ memory)
- ⚠️ Slower than API models
- ⚠️ Limited context length (2048 tokens)
- ⚠️ Model may need to be downloaded/configured from GitHub

The implementation maintains compatibility with the original evaluation pipeline, so you can directly compare results between LLaDA 2.0, Dream 7B, and GPT-4.1-mini.

## Where to Change Things

### Model Path
- **Location**: `text_simulation/configs/llada_config.yaml`
- **Parameter**: `model_name`
- **Update when**: LLaDA model is released on HuggingFace with a different path

### Generation Parameters
- **Location**: `text_simulation/configs/llada_config.yaml`
- **Parameters**: `steps`, `remasking_strategy`, `guidance_scale`, `temperature`
- **Update when**: You want to experiment with different generation settings

### Context Length
- **Location**: `text_simulation/configs/llada_config.yaml`
- **Parameter**: `max_context_length`
- **Update when**: Model supports longer contexts or you need to adjust truncation

### Generation Method
- **Location**: `text_simulation/llm_helper_llada.py`
- **Function**: `_get_llada_response_direct()`
- **Update when**: LLaDA API changes or you need to use different generation methods

### Output Directory
- **Location**: `text_simulation/configs/llada_config.yaml`
- **Parameter**: `output_folder_dir`
- **Update when**: You want to change where results are saved


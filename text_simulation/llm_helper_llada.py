"""
LLaDA 2.0 LLM Helper for Digital Twin Simulation

This module provides support for LLaDA 2.0 (Large Language Diffusion Model) as an alternative
to OpenAI/Gemini API-based models or Dream. LLaDA is a masked diffusion language model that
uses diffusion-based generation instead of autoregressive generation.

References:
- Paper: https://arxiv.org/abs/2502.09992
- GitHub: https://github.com/ML-GSAI/LLaDA
- Demo: https://ml-gsai.github.io/LLaDA-demo/
- HuggingFace: https://huggingface.co/papers/2502.09992
"""

import os
import torch
from typing import Dict, Optional, Union, Callable, List, Tuple
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import asyncio
from tqdm.asyncio import tqdm_asyncio
import threading

load_dotenv()

# System instruction for LLaDA (same as other models)
LLADA_SYSTEM_INSTRUCTION = """You are an AI assistant. Your task is to answer the 'New Survey Question' as if you are the person described in the 'Persona Profile' (which consists of their past survey responses). 
Adhere to the persona by being consistent with their previous answers and stated characteristics. 
Follow all instructions provided for the new question carefully regarding the format of your answer."""

# Global model and tokenizer (loaded once, reused for all requests)
_model_instance = None
_tokenizer_instance = None
_model_lock = threading.Lock()


def _load_llada_model(model_path: str = "ML-GSAI/LLaDA-8B-Instruct", device: str = "cuda"):
    """
    Load LLaDA model and tokenizer. Uses singleton pattern to load once.
    
    Args:
        model_path: HuggingFace model path (default: "ML-GSAI/LLaDA-8B-Instruct")
        device: Device to load model on (default: "cuda")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    global _model_instance, _tokenizer_instance
    
    with _model_lock:
        if _model_instance is None or _tokenizer_instance is None:
            print(f"Loading LLaDA model from {model_path}...")
            print("Note: This requires transformers and torch, and at least 20GB GPU memory")
            
            # Load tokenizer
            _tokenizer_instance = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # Load model
            _model_instance = AutoModel.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            _model_instance = _model_instance.to(device).eval()
            
            print(f"LLaDA model loaded successfully on {device}")
    
    return _model_instance, _tokenizer_instance


class LLaDALLMConfig:
    """
    Configuration for LLaDA 2.0 model.
    
    Note: LLaDA uses masked diffusion-based generation with different parameters
    than standard autoregressive models.
    """
    def __init__(
        self,
        model_name: str = "ML-GSAI/LLaDA-8B-Instruct",
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        steps: int = 512,  # Diffusion timesteps
        remasking_strategy: str = "random",  # Remasking strategy for masked diffusion
        guidance_scale: float = 1.0,  # Classifier-free guidance scale
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        system_instruction: Optional[str] = None,
        max_retries: int = 10,
        max_concurrent_requests: int = 1,  # Lower for local model (GPU memory limited)
        device: str = "cuda",
        verification_callback: Optional[Callable[..., bool]] = None,
        verification_callback_args: Optional[Dict] = None,
        max_context_length: int = 2048,  # LLaDA's context limit
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.remasking_strategy = remasking_strategy
        self.guidance_scale = guidance_scale
        self.top_p = top_p
        self.top_k = top_k
        self.system_instruction = system_instruction or LLADA_SYSTEM_INSTRUCTION
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self.device = device
        self.verification_callback = verification_callback
        self.verification_callback_args = verification_callback_args if verification_callback_args is not None else {}
        self.max_context_length = max_context_length


def _truncate_prompt_if_needed(prompt: str, tokenizer, max_length: int = 2048) -> str:
    """
    Truncate prompt if it exceeds context length.
    LLaDA has a context limit (input + output).
    
    Args:
        prompt: Original prompt text
        tokenizer: Tokenizer to count tokens
        max_length: Maximum context length
    
    Returns:
        Truncated prompt if needed
    """
    # Tokenize to check length
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    if len(tokens) <= max_length:
        return prompt
    
    # Truncate from the beginning (keep the end which has the question)
    # Reserve some tokens for output
    reserved_for_output = 512
    max_input_tokens = max_length - reserved_for_output
    
    if len(tokens) > max_input_tokens:
        truncated_tokens = tokens[-max_input_tokens:]
        truncated_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        print(f"Warning: Prompt truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
        return truncated_prompt
    
    return prompt


def _get_llada_response_direct(prompt: str, config: LLaDALLMConfig) -> Dict[str, Union[str, Dict]]:
    """
    Get response from LLaDA model using masked diffusion generation.
    
    Args:
        prompt: Input prompt text
        config: LLaDALLMConfig with generation parameters
    
    Returns:
        Dictionary with response_text and usage_details
    """
    try:
        # Load model (singleton pattern)
        model, tokenizer = _load_llada_model(config.model_name, config.device)
        
        # Truncate prompt if needed
        prompt = _truncate_prompt_if_needed(prompt, tokenizer, config.max_context_length)
        
        # Format messages with chat template
        # LLaDA uses chat template similar to other instruct models
        messages = [
            {"role": "system", "content": config.system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            return_dict=True, 
            add_generation_prompt=True
        )
        
        input_ids = inputs.input_ids.to(device=config.device)
        attention_mask = inputs.attention_mask.to(device=config.device)
        
        # Generate using masked diffusion
        # LLaDA uses diffusion_generate or similar method
        with torch.no_grad():
            # Check if model has diffusion_generate method (similar to Dream)
            if hasattr(model, 'diffusion_generate'):
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=config.steps,
                    temperature=config.temperature,
                    top_p=config.top_p if config.top_p else None,
                    top_k=config.top_k if config.top_k else None,
                    remasking_strategy=config.remasking_strategy,
                    guidance_scale=config.guidance_scale,
                )
            elif hasattr(model, 'generate'):
                # Fallback to standard generate if diffusion_generate not available
                # LLaDA might use a different API
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p if config.top_p else None,
                    top_k=config.top_k if config.top_k else None,
                    do_sample=True,
                    return_dict_in_generate=True,
                )
            else:
                raise AttributeError("Model does not have generate or diffusion_generate method")
        
        # Decode the generated tokens
        if hasattr(output, 'sequences'):
            sequences = output.sequences
        elif isinstance(output, torch.Tensor):
            sequences = output
        else:
            sequences = output[0] if isinstance(output, (list, tuple)) else output
        
        generations = [
            tokenizer.decode(g[len(p):].tolist(), skip_special_tokens=True)
            for p, g in zip(input_ids, sequences)
        ]
        
        response_text = generations[0].split(tokenizer.eos_token)[0] if tokenizer.eos_token else generations[0]
        
        # Estimate token usage
        input_token_count = input_ids.shape[1]
        if isinstance(sequences, torch.Tensor):
            output_token_count = sequences.shape[1] - input_token_count
        else:
            output_token_count = len(sequences[0]) - input_token_count if isinstance(sequences, list) else 0
        
        usage_details = {
            "prompt_token_count": input_token_count,
            "completion_token_count": output_token_count,
            "total_token_count": input_token_count + output_token_count,
            "steps": config.steps,
            "remasking_strategy": config.remasking_strategy,
            "guidance_scale": config.guidance_scale
        }
        
        return {"response_text": response_text, "usage_details": usage_details}
        
    except Exception as e:
        return {"error": f"LLaDA generation failed: {str(e)}", "response_text": "", "usage_details": {}}


async def get_llada_response_with_retry(
    prompt: str, 
    config: LLaDALLMConfig
) -> Dict[str, Union[str, Dict]]:
    """
    Get LLaDA response with retry logic (runs in thread pool since model is synchronous).
    
    Args:
        prompt: Input prompt text
        config: LLaDALLMConfig
    
    Returns:
        Dictionary with response or error
    """
    # Run in thread pool since model inference is synchronous
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _get_llada_response_direct(prompt, config)
    )
    return result


async def _process_single_prompt_attempt_with_verification_llada(
    prompt_id: str,
    prompt_text: str,
    config: LLaDALLMConfig,
    semaphore: asyncio.Semaphore
):
    """
    Process a single prompt with LLaDA model, including verification.
    
    Args:
        prompt_id: Unique identifier for the prompt
        prompt_text: The prompt text
        config: LLaDALLMConfig
        semaphore: Semaphore for concurrency control
    
    Returns:
        Tuple of (prompt_id, response_data)
    """
    async with semaphore:  # Manage concurrency
        last_exception_details = None
        for attempt in range(config.max_retries):
            llm_response_data = None
            try:
                # Step 1: Get LLaDA response
                llm_response_data = await get_llada_response_with_retry(prompt_text, config)
                
                if "error" in llm_response_data and llm_response_data["error"]:
                    last_exception_details = llm_response_data
                    if attempt == config.max_retries - 1:
                        return prompt_id, last_exception_details
                    await asyncio.sleep(min(2 * 2 ** attempt, 30))  # Exponential backoff
                    continue
                
                # Step 2: Perform verification if callback is provided
                if config.verification_callback:
                    verified = await asyncio.to_thread(
                        config.verification_callback,
                        prompt_id,
                        llm_response_data,
                        prompt_text,
                        **config.verification_callback_args
                    )
                    if not verified:
                        last_exception_details = {
                            "error": f"Verification failed on attempt {attempt + 1}",
                            "prompt_id": prompt_id,
                            "llm_response_data": llm_response_data
                        }
                        if attempt == config.max_retries - 1:
                            return prompt_id, last_exception_details
                        await asyncio.sleep(min(2 * 2 ** attempt, 30))
                        continue
                
                # Success
                return prompt_id, llm_response_data
                
            except Exception as e:
                last_exception_details = {
                    "error": f"Unexpected error: {str(e)}",
                    "prompt_id": prompt_id
                }
                if attempt == config.max_retries - 1:
                    return prompt_id, last_exception_details
                await asyncio.sleep(min(2 * 2 ** attempt, 30))
                continue
        
        # Fallback
        return prompt_id, last_exception_details if last_exception_details else {
            "error": f"Exhausted all {config.max_retries} retries for {prompt_id}"
        }


async def process_prompts_batch_llada(
    prompts: List[Tuple[str, str]],
    config: LLaDALLMConfig,
    desc: Optional[str] = "Processing LLaDA prompts and verifying"
) -> Dict[str, Dict[str, Union[str, Dict]]]:
    """
    Process a batch of prompts with LLaDA model.
    
    Args:
        prompts: List of (prompt_id, prompt_text) tuples
        config: LLaDALLMConfig
        desc: Description for progress bar
    
    Returns:
        Dictionary mapping prompt_id to response data
    """
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    results = {}
    
    tasks = [
        _process_single_prompt_attempt_with_verification_llada(pid, p_text, config, semaphore)
        for pid, p_text in prompts
    ]
    
    for future in tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        prompt_id, response_data = await future
        results[prompt_id] = response_data
    
    return results


# Example usage
if __name__ == "__main__":
    async def mock_verification_callback(prompt_id, llm_response_data, original_prompt_text, **kwargs):
        print(f"  (Mock Verify for {prompt_id}): Response length: {len(llm_response_data.get('response_text', ''))}")
        return True
    
    async def main():
        llada_config = LLaDALLMConfig(
            model_name="ML-GSAI/LLaDA-8B-Instruct",
            temperature=0.2,
            max_new_tokens=512,
            steps=512,
            remasking_strategy="random",
            guidance_scale=1.0,
            max_retries=3,
            max_concurrent_requests=1,  # Start with 1 for testing
            verification_callback=mock_verification_callback,
            verification_callback_args={"test": True}
        )
        
        prompts = [
            ("p1", "What is the capital of France? Answer concisely."),
            ("p2", "What is 2+2? Answer concisely."),
        ]
        
        print("\nProcessing LLaDA prompts...")
        results = await process_prompts_batch_llada(prompts, llada_config, desc="LLaDA Calls")
        
        for pid, resp in results.items():
            if "error" in resp and resp["error"]:
                print(f"LLaDA - Prompt {pid} ERROR: {resp['error']}")
            else:
                print(f"LLaDA - Prompt {pid} OK. Response: '{resp.get('response_text', '')[:50]}...'")
    
    asyncio.run(main())


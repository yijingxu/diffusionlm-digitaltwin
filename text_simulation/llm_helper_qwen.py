"""
Qwen2.5-7B-Instruct LLM Helper for Digital Twin Simulation

This module provides support for Qwen2.5-7B-Instruct as an alternative
to OpenAI/Gemini API-based models or Dream. Qwen is a standard autoregressive
language model that runs locally.

References:
- Model: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- GitHub: https://github.com/QwenLM/Qwen
"""

import os
import torch
from typing import Dict, Optional, Union, Callable, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import asyncio
from tqdm.asyncio import tqdm_asyncio
import threading

load_dotenv()

# Default system instruction for Qwen (can be overridden in config)
QWEN_SYSTEM_INSTRUCTION = """You are an AI assistant that predicts human behavior based on demographic characteristics and psychographic traits. 
Given demographic information and psychographic measures (Need for Cognition, Spendthrift/Tightwad, Maximization, Minimalism), predict how a person would respond to behavioral questions. 
Respond with only a single number corresponding to the option they would choose."""

# Global model and tokenizer (loaded once, reused for all requests)
_model_instance = None
_tokenizer_instance = None
_model_lock = threading.Lock()


def _load_qwen_model(model_path: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
    """
    Load Qwen model and tokenizer. Uses singleton pattern to load once.
    
    Args:
        model_path: HuggingFace model path
        device: Device to load model on (default: "cuda")
    
    Returns:
        Tuple of (model, tokenizer)
    
    Raises:
        Exception: If model loading fails
    """
    global _model_instance, _tokenizer_instance
    
    with _model_lock:
        if _model_instance is None or _tokenizer_instance is None:
            print(f"Loading Qwen model from {model_path}...")
            print("Note: This requires transformers and torch, and at least 14GB GPU memory")
            
            try:
                # Load tokenizer
                _tokenizer_instance = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True
                )
                print(f"Tokenizer loaded successfully")
                
                # Load model
                _model_instance = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True
                )
                print(f"Model loaded from HuggingFace, moving to {device}...")
                _model_instance = _model_instance.to(device).eval()
                
                print(f"Qwen model loaded successfully on {device}")
            except Exception as e:
                # Clear instances on failure so we don't keep retrying with broken state
                _model_instance = None
                _tokenizer_instance = None
                error_msg = f"Failed to load Qwen model from {model_path}: {str(e)}"
                print(f"\n{'='*80}")
                print(f"ERROR: {error_msg}")
                print(f"Error type: {type(e).__name__}")
                print(f"\nPossible causes:")
                print(f"  1. Model path '{model_path}' may be incorrect")
                print(f"  2. Model may not exist on HuggingFace")
                print(f"  3. Insufficient GPU memory (need at least 14GB)")
                print(f"  4. Missing dependencies or incompatible versions")
                print(f"  5. Network issues preventing model download")
                print(f"{'='*80}\n")
                raise RuntimeError(error_msg) from e
    
    if _model_instance is None or _tokenizer_instance is None:
        raise RuntimeError("Model or tokenizer is None after loading attempt")
    
    return _model_instance, _tokenizer_instance


class QwenLLMConfig:
    """
    Configuration for Qwen2.5-7B-Instruct model.
    
    Note: Qwen uses standard autoregressive generation.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        system_instruction: Optional[str] = None,
        max_retries: int = 10,
        max_concurrent_requests: int = 1,  # Lower for local model (GPU memory limited)
        device: str = "cuda",
        verification_callback: Optional[Callable[..., bool]] = None,
        verification_callback_args: Optional[Dict] = None,
        max_context_length: int = 32768,  # Qwen2.5 has 32K context
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.system_instruction = system_instruction or QWEN_SYSTEM_INSTRUCTION
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self.device = device
        self.verification_callback = verification_callback
        self.verification_callback_args = verification_callback_args if verification_callback_args is not None else {}
        self.max_context_length = max_context_length


def _truncate_prompt_if_needed(prompt: str, tokenizer, max_length: int = 32768) -> str:
    """
    Truncate prompt if it exceeds context length.
    Qwen2.5 has a 32K token context limit (input + output).
    
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


def _get_qwen_response_direct(prompt: str, config: QwenLLMConfig) -> Dict[str, Union[str, Dict]]:
    """
    Get response from Qwen model using standard autoregressive generation.
    
    Args:
        prompt: Input prompt text
        config: QwenLLMConfig with generation parameters
    
    Returns:
        Dictionary with response_text and usage_details
    """
    try:
        # Load model (singleton pattern)
        model, tokenizer = _load_qwen_model(config.model_name, config.device)
        
        # Truncate prompt if needed
        prompt = _truncate_prompt_if_needed(prompt, tokenizer, config.max_context_length)
        
        # Format messages with chat template
        # Qwen uses chat template similar to other instruct models
        messages = [
            {"role": "system", "content": config.system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(device=config.device)
        
        # Generate using standard autoregressive generation
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p if config.top_p else None,
                top_k=config.top_k if config.top_k else None,
                do_sample=config.temperature > 0.0,  # Only sample if temperature > 0
            )
        
        # Decode the generated tokens (skip the input tokens)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Estimate token usage
        input_token_count = model_inputs.input_ids.shape[1]
        output_token_count = len(generated_ids[0])
        
        usage_details = {
            "prompt_token_count": input_token_count,
            "completion_token_count": output_token_count,
            "total_token_count": input_token_count + output_token_count,
        }
        
        return {"response_text": response_text, "usage_details": usage_details}
        
    except Exception as e:
        return {"error": f"Qwen generation failed: {str(e)}", "response_text": "", "usage_details": {}}


async def get_qwen_response_with_retry(
    prompt: str, 
    config: QwenLLMConfig
) -> Dict[str, Union[str, Dict]]:
    """
    Get Qwen response with retry logic (runs in thread pool since model is synchronous).
    
    Args:
        prompt: Input prompt text
        config: QwenLLMConfig
    
    Returns:
        Dictionary with response or error
    """
    # Run in thread pool since model inference is synchronous
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _get_qwen_response_direct(prompt, config)
    )
    return result


async def _process_single_prompt_attempt_with_verification_qwen(
    prompt_id: str,
    prompt_text: str,
    config: QwenLLMConfig,
    semaphore: asyncio.Semaphore
):
    """
    Process a single prompt with Qwen model, including verification.
    
    Args:
        prompt_id: Unique identifier for the prompt
        prompt_text: The prompt text
        config: QwenLLMConfig
        semaphore: Semaphore for concurrency control
    
    Returns:
        Tuple of (prompt_id, response_data)
    """
    async with semaphore:  # Manage concurrency
        last_exception_details = None
        for attempt in range(config.max_retries):
            llm_response_data = None
            try:
                # Step 1: Get Qwen response
                llm_response_data = await get_qwen_response_with_retry(prompt_text, config)
                
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


async def process_prompts_batch_qwen(
    prompts: List[Tuple[str, str]],
    config: QwenLLMConfig,
    desc: Optional[str] = "Processing Qwen prompts and verifying"
) -> Dict[str, Dict[str, Union[str, Dict]]]:
    """
    Process a batch of prompts with Qwen model.
    
    Args:
        prompts: List of (prompt_id, prompt_text) tuples
        config: QwenLLMConfig
        desc: Description for progress bar
    
    Returns:
        Dictionary mapping prompt_id to response data
    """
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    results = {}
    
    tasks = [
        _process_single_prompt_attempt_with_verification_qwen(pid, p_text, config, semaphore)
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
        qwen_config = QwenLLMConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.2,
            max_new_tokens=512,
            max_retries=3,
            max_concurrent_requests=1,  # Start with 1 for testing
            verification_callback=mock_verification_callback,
            verification_callback_args={"test": True}
        )
        
        prompts = [
            ("p1", "What is the capital of France? Answer concisely."),
            ("p2", "What is 2+2? Answer concisely."),
        ]
        
        print("\nProcessing Qwen prompts...")
        results = await process_prompts_batch_qwen(prompts, qwen_config, desc="Qwen Calls")
        
        for pid, resp in results.items():
            if "error" in resp and resp["error"]:
                print(f"Qwen - Prompt {pid} ERROR: {resp['error']}")
            else:
                print(f"Qwen - Prompt {pid} OK. Response: '{resp.get('response_text', '')[:50]}...'")
    
    asyncio.run(main())


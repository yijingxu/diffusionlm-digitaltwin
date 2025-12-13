"""
Dream 7B LLM Helper for Digital Twin Simulation

This module provides support for Dream 7B (diffusion language model) as an alternative
to OpenAI/Gemini API-based models. Dream 7B is a local model that uses diffusion-based
generation instead of autoregressive generation.

References:
- Model: https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
- GitHub: https://github.com/DreamLM/Dream
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

# System instruction for Dream (same as other models)
DREAM_SYSTEM_INSTRUCTION = """You are an AI assistant. Your task is to answer the 'New Survey Question' as if you are the person described in the 'Persona Profile' (which consists of their past survey responses). 
Adhere to the persona by being consistent with their previous answers and stated characteristics. 
Follow all instructions provided for the new question carefully regarding the format of your answer."""

# Global model and tokenizer (loaded once, reused for all requests)
_model_instance = None
_tokenizer_instance = None
_model_lock = threading.Lock()


def _load_dream_model(model_path: str = "Dream-org/Dream-v0-Instruct-7B", device: str = "cuda"):
    """
    Load Dream model and tokenizer. Uses singleton pattern to load once.
    
    Args:
        model_path: HuggingFace model path
        device: Device to load model on (default: "cuda")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    global _model_instance, _tokenizer_instance
    
    with _model_lock:
        if _model_instance is None or _tokenizer_instance is None:
            print(f"Loading Dream model from {model_path}...")
            print("Note: This requires transformers==4.46.2, torch==2.5.1, and at least 20GB GPU memory")
            
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
            
            print(f"Dream model loaded successfully on {device}")
    
    return _model_instance, _tokenizer_instance


class DreamLLMConfig:
    """
    Configuration for Dream 7B model.
    
    Note: Dream uses diffusion-based generation with different parameters
    than standard autoregressive models.
    """
    def __init__(
        self,
        model_name: str = "Dream-org/Dream-v0-Instruct-7B",
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        steps: int = 512,  # Diffusion timesteps
        alg: str = "entropy",  # Remasking strategy: "origin", "maskgit_plus", "topk_margin", "entropy"
        alg_temp: float = 0.0,  # Randomness for confidence-based strategies
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        system_instruction: Optional[str] = None,
        max_retries: int = 10,
        max_concurrent_requests: int = 1,  # Lower for local model (GPU memory limited)
        device: str = "cuda",
        verification_callback: Optional[Callable[..., bool]] = None,
        verification_callback_args: Optional[Dict] = None,
        max_context_length: int = 2048,  # Dream's context limit
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.alg = alg
        self.alg_temp = alg_temp
        self.top_p = top_p
        self.top_k = top_k
        self.system_instruction = system_instruction or DREAM_SYSTEM_INSTRUCTION
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self.device = device
        self.verification_callback = verification_callback
        self.verification_callback_args = verification_callback_args if verification_callback_args is not None else {}
        self.max_context_length = max_context_length


def _truncate_prompt_if_needed(prompt: str, tokenizer, max_length: int = 2048) -> str:
    """
    Truncate prompt if it exceeds context length.
    Dream has a 2048 token context limit (input + output).
    
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


def _get_dream_response_direct(prompt: str, config: DreamLLMConfig) -> Dict[str, Union[str, Dict]]:
    """
    Get response from Dream model using diffusion generation.
    
    Args:
        prompt: Input prompt text
        config: DreamLLMConfig with generation parameters
    
    Returns:
        Dictionary with response_text and usage_details
    """
    try:
        # Load model (singleton pattern)
        model, tokenizer = _load_dream_model(config.model_name, config.device)
        
        # Truncate prompt if needed (Dream has 2048 token limit)
        prompt = _truncate_prompt_if_needed(prompt, tokenizer, config.max_context_length)
        
        # Format messages with chat template
        # Dream uses chat template similar to other instruct models
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
        
        # Generate using diffusion
        with torch.no_grad():
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                output_history=False,  # Set to True if you want intermediate steps
                return_dict_in_generate=True,
                steps=config.steps,
                temperature=config.temperature,
                top_p=config.top_p if config.top_p else None,
                top_k=config.top_k if config.top_k else None,
                alg=config.alg,
                alg_temp=config.alg_temp if config.alg_temp else None,
            )
        
        # Decode the generated tokens
        generations = [
            tokenizer.decode(g[len(p):].tolist(), skip_special_tokens=True)
            for p, g in zip(input_ids, output.sequences)
        ]
        
        response_text = generations[0].split(tokenizer.eos_token)[0] if tokenizer.eos_token else generations[0]
        
        # Estimate token usage (Dream doesn't provide detailed usage stats)
        input_token_count = input_ids.shape[1]
        output_token_count = output.sequences.shape[1] - input_token_count
        
        usage_details = {
            "prompt_token_count": input_token_count,
            "completion_token_count": output_token_count,
            "total_token_count": input_token_count + output_token_count,
            "steps": config.steps,
            "alg": config.alg
        }
        
        return {"response_text": response_text, "usage_details": usage_details}
        
    except Exception as e:
        return {"error": f"Dream generation failed: {str(e)}", "response_text": "", "usage_details": {}}


async def get_dream_response_with_retry(
    prompt: str, 
    config: DreamLLMConfig
) -> Dict[str, Union[str, Dict]]:
    """
    Get Dream response with retry logic (runs in thread pool since model is synchronous).
    
    Args:
        prompt: Input prompt text
        config: DreamLLMConfig
    
    Returns:
        Dictionary with response or error
    """
    # Run in thread pool since model inference is synchronous
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _get_dream_response_direct(prompt, config)
    )
    return result


async def _process_single_prompt_attempt_with_verification_dream(
    prompt_id: str,
    prompt_text: str,
    config: DreamLLMConfig,
    semaphore: asyncio.Semaphore
):
    """
    Process a single prompt with Dream model, including verification.
    
    Args:
        prompt_id: Unique identifier for the prompt
        prompt_text: The prompt text
        config: DreamLLMConfig
        semaphore: Semaphore for concurrency control
    
    Returns:
        Tuple of (prompt_id, response_data)
    """
    async with semaphore:  # Manage concurrency
        last_exception_details = None
        for attempt in range(config.max_retries):
            llm_response_data = None
            try:
                # Step 1: Get Dream response
                llm_response_data = await get_dream_response_with_retry(prompt_text, config)
                
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


async def process_prompts_batch_dream(
    prompts: List[Tuple[str, str]],
    config: DreamLLMConfig,
    desc: Optional[str] = "Processing Dream prompts and verifying"
) -> Dict[str, Dict[str, Union[str, Dict]]]:
    """
    Process a batch of prompts with Dream model.
    
    Args:
        prompts: List of (prompt_id, prompt_text) tuples
        config: DreamLLMConfig
        desc: Description for progress bar
    
    Returns:
        Dictionary mapping prompt_id to response data
    """
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    results = {}
    
    tasks = [
        _process_single_prompt_attempt_with_verification_dream(pid, p_text, config, semaphore)
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
        dream_config = DreamLLMConfig(
            model_name="Dream-org/Dream-v0-Instruct-7B",
            temperature=0.2,
            max_new_tokens=512,
            steps=512,
            alg="entropy",
            max_retries=3,
            max_concurrent_requests=1,  # Start with 1 for testing
            verification_callback=mock_verification_callback,
            verification_callback_args={"test": True}
        )
        
        prompts = [
            ("p1", "What is the capital of France? Answer concisely."),
            ("p2", "What is 2+2? Answer concisely."),
        ]
        
        print("\nProcessing Dream prompts...")
        results = await process_prompts_batch_dream(prompts, dream_config, desc="Dream Calls")
        
        for pid, resp in results.items():
            if "error" in resp and resp["error"]:
                print(f"Dream - Prompt {pid} ERROR: {resp['error']}")
            else:
                print(f"Dream - Prompt {pid} OK. Response: '{resp.get('response_text', '')[:50]}...'")
    
    asyncio.run(main())


"""
Run LLM Simulations for Vaccine Prediction using Dream 7B Model

This script uses demographic-only data to predict vaccine acceptance likelihood.
It loads a clean dataset with demographics and uses Dream 7B to predict responses.

Usage:
    python run_LLM_simulations_dream_vaccine.py --config text_simulation/configs/dream_config.yaml --dataset ./data/demographic_vaccine_dataset.json
"""

import os
import json
import argparse
import yaml
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv
from llm_helper_dream import DreamLLMConfig, process_prompts_batch_dream
from vaccine_prediction_helpers import (
    create_vaccine_prompt,
    parse_vaccine_response,
    validate_vaccine_response
)
from datetime import datetime

load_dotenv()


# Updated system instruction for vaccine prediction
VACCINE_PREDICTION_SYSTEM_INSTRUCTION = """You are an AI assistant that predicts human behavior based on demographic characteristics. 
Given demographic information about a person, predict how they would respond to a specific question 
about vaccine acceptance. Respond with only a single number from 1-4, where:
1 = I would definitely not take the vaccine
2 = I would probably not take the vaccine
3 = I would probably take the vaccine
4 = I would definitely take the vaccine"""


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data


def load_dataset(dataset_path: str) -> list:
    """Load the demographic-vaccine dataset."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def get_output_path(base_output_dir: str, persona_id: str) -> str:
    """Get output file path for a persona prediction."""
    persona_output_folder = os.path.join(base_output_dir, persona_id)
    os.makedirs(persona_output_folder, exist_ok=True)
    output_filename = f"{persona_id}_vaccine_prediction.json"
    return os.path.join(persona_output_folder, output_filename)


def save_and_verify_vaccine_callback(
    prompt_id: str, 
    llm_response_data: dict, 
    original_prompt_text: str, 
    **kwargs
) -> bool:
    """
    Saves the Dream LLM response and verifies it contains a valid vaccine prediction (1-4).
    
    Expected kwargs: base_output_dir, persona_to_entry
    """
    base_output_dir = kwargs.get("base_output_dir")
    persona_to_entry = kwargs.get("persona_to_entry", {})
    
    if not base_output_dir:
        print(f"Error for {prompt_id}: Missing base_output_dir in verification_callback_args.")
        return False
    
    dataset_entry = persona_to_entry.get(prompt_id)
    if not dataset_entry:
        print(f"Error for {prompt_id}: No dataset entry found for this persona.")
        return False
    
    persona_id = prompt_id
    output_path = get_output_path(base_output_dir, persona_id)
    
    # Parse the response
    response_text = llm_response_data.get("response_text", "")
    predicted_likelihood = parse_vaccine_response(response_text)
    
    # Save prediction
    output_json_data = {
        "persona_id": persona_id,
        "demographic_description": dataset_entry.get("demographic_description", ""),
        "vaccine_question": {
            "question_id": dataset_entry["vaccine_question"]["question_id"],
            "question_text": dataset_entry["vaccine_question"]["question_text"]
        },
        "predicted_likelihood": predicted_likelihood,
        "predicted_likelihood_text": {
            1: "I would definitely not take the vaccine",
            2: "I would probably not take the vaccine",
            3: "I would probably take the vaccine",
            4: "I would definitely take the vaccine"
        }.get(predicted_likelihood, None),
        "actual_answer": dataset_entry["vaccine_question"]["actual_answer"],
        "actual_answer_text": dataset_entry["vaccine_question"]["actual_answer_text"],
        "response_text": response_text,
        "prompt_text": original_prompt_text,
        "usage_details": llm_response_data.get("usage_details", {}),
        "llm_call_error": llm_response_data.get("error"),
        "model": "Dream-7B"
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing output file {output_path} for {prompt_id}: {e}")
        return False
    
    # If the LLM call itself had an error, no point in verifying
    if "error" in llm_response_data and llm_response_data["error"]:
        return False
    
    # Verify the response contains a valid prediction
    is_valid = validate_vaccine_response(response_text)
    if not is_valid:
        print(f"Warning: Invalid vaccine response for {prompt_id}: {response_text[:100]}")
    
    return is_valid


async def run_vaccine_predictions(
    dataset_path: str,
    base_output_dir: str,
    dream_config_params: dict,
    num_workers: int,
    max_retries_for_sequence: int,
    force_regenerate: bool,
    max_personas=None,
    start_index=None,
    batch_size=None
):
    """
    Run vaccine predictions using Dream 7B model.
    
    Args:
        dataset_path: Path to demographic_vaccine_dataset.json
        base_output_dir: Directory to save outputs
        dream_config_params: Dictionary of Dream configuration parameters
        num_workers: Number of concurrent requests (typically 1 for local model)
        max_retries_for_sequence: Max retries for LLM call + verification
        force_regenerate: Whether to regenerate existing outputs
        max_personas: Maximum number of personas to process (None for all)
        start_index: Starting index in sorted list (0-based, for distributed processing)
        batch_size: Number of personas to process from start_index (None for all remaining)
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} personas from dataset.")
    
    # Prepare arguments for the verification callback
    # We'll add dataset entries after creating prompts
    verification_args = {
        "base_output_dir": base_output_dir,
    }
    
    # Use custom system instruction for vaccine prediction
    system_instruction = dream_config_params.get(
        'system_instruction', 
        VACCINE_PREDICTION_SYSTEM_INSTRUCTION
    )
    
    # Create Dream config
    dream_config = DreamLLMConfig(
        model_name=dream_config_params.get('model_name', 'Dream-org/Dream-v0-Instruct-7B'),
        temperature=dream_config_params.get('temperature', 0.2),
        max_new_tokens=dream_config_params.get('max_new_tokens', 512),
        steps=dream_config_params.get('steps', 512),
        alg=dream_config_params.get('alg', 'entropy'),
        alg_temp=dream_config_params.get('alg_temp', 0.0),
        top_p=dream_config_params.get('top_p', 0.95),
        top_k=dream_config_params.get('top_k', None),
        system_instruction=system_instruction,
        max_retries=max_retries_for_sequence,
        max_concurrent_requests=num_workers,
        device=dream_config_params.get('device', 'cuda'),
        verification_callback=save_and_verify_vaccine_callback,
        verification_callback_args=verification_args,
        max_context_length=dream_config_params.get('max_context_length', 2048)
    )
    
    # Create output directory
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Prepare prompts
    prompts_to_process = []
    skipped_due_to_existing_count = 0
    
    # Apply start_index and batch_size filtering
    if start_index is not None:
        if start_index < 0:
            start_index = 0
        if start_index >= len(dataset):
            print(f"Warning: start_index {start_index} is >= dataset size {len(dataset)}. No entries to process.")
            return
        dataset = dataset[start_index:]
        print(f"Starting from index {start_index}")
    
    if batch_size is not None and batch_size > 0:
        dataset = dataset[:batch_size]
        print(f"Processing batch of {batch_size} entries")
    elif max_personas is not None and max_personas > 0:
        dataset = dataset[:max_personas]
        print(f"Limiting processing to {max_personas} personas")
    
    # Check for existing outputs and create prompts
    for entry in dataset:
        persona_id = entry["persona_id"]
        output_path = get_output_path(base_output_dir, persona_id)
        
        if os.path.exists(output_path) and not force_regenerate:
            # Check if existing output is valid
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if existing_data.get("predicted_likelihood") is not None:
                    skipped_due_to_existing_count += 1
                    continue
            except:
                pass  # If we can't read it, regenerate
        
        # Create prompt from demographic description
        demographic_description = entry["demographic_description"]
        prompt_text = create_vaccine_prompt(demographic_description)
        
        prompts_to_process.append({
            "persona_id": persona_id,
            "prompt_text": prompt_text,
            "dataset_entry": entry
        })
    
    if skipped_due_to_existing_count > 0:
        print(f"Skipped {skipped_due_to_existing_count} entries as their predictions already exist.")
    
    if not prompts_to_process:
        print("No new entries require Dream processing or re-processing.")
        if skipped_due_to_existing_count == 0:
            print("No valid entries found or processed in this run.")
        return
    
    print(f"Found {len(prompts_to_process)} personas to process with Dream 7B (up to {dream_config.max_concurrent_requests} concurrent requests).")
    print(f"Note: Dream is a local model. Processing may be slower than API-based models.")
    
    # Create mapping of persona_id to dataset_entry and prompt_text
    persona_to_entry = {p["persona_id"]: p["dataset_entry"] for p in prompts_to_process}
    persona_to_prompt = {p["persona_id"]: p["prompt_text"] for p in prompts_to_process}
    verification_args["persona_to_entry"] = persona_to_entry
    verification_args["persona_to_prompt"] = persona_to_prompt
    
    # Convert to format expected by process_prompts_batch_dream
    prompts_for_batch = [(p["persona_id"], p["prompt_text"]) for p in prompts_to_process]
    
    # Process prompts with Dream
    final_results = await process_prompts_batch_dream(
        prompts_for_batch,
        dream_config,
        desc="Dream 7B Vaccine Predictions"
    )
    
    # Re-save results with proper dataset_entry (callback will be called again but that's okay, it will overwrite correctly)
    for prompt_id, result_data in final_results.items():
        if prompt_id in persona_to_entry:
            verification_args_current = {
                "base_output_dir": base_output_dir,
                "dataset_entry": persona_to_entry[prompt_id]
            }
            original_prompt = persona_to_prompt.get(prompt_id, "")
            # This will properly save with all the dataset information
            save_and_verify_vaccine_callback(prompt_id, result_data, original_prompt, **verification_args_current)
    
    # Count successes and failures
    successful_count = 0
    failed_count = 0
    
    for prompt_id, result_data in final_results.items():
        if "error" in result_data and result_data["error"]:
            failed_count += 1
            print(f"FINAL FAILURE for {prompt_id}: {result_data['error']}")
        else:
            # Check if prediction was valid
            response_text = result_data.get("response_text", "")
            if validate_vaccine_response(response_text):
                successful_count += 1
            else:
                failed_count += 1
                print(f"INVALID PREDICTION for {prompt_id}: {response_text[:100]}")
    
    print(f"\nDream 7B vaccine prediction run complete.")
    print(f"  Successfully processed and verified: {successful_count}")
    print(f"  Failed permanently after all retries: {failed_count}")
    if skipped_due_to_existing_count > 0:
        print(f"  Skipped (already existing): {skipped_due_to_existing_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Dream 7B vaccine predictions with demographic-only data."
    )
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--dataset",
        default="./data/demographic_vaccine_dataset.json",
        help="Path to demographic_vaccine_dataset.json"
    )
    parser.add_argument(
        "--max_personas", 
        type=int, 
        help="Maximum number of personas to process (DEPRECATED: use --start_index and --batch_size)"
    )
    parser.add_argument(
        "--start_index", 
        type=int, 
        help="Starting index in dataset (0-based, for distributed processing)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Number of personas to process from start_index"
    )
    
    args = parser.parse_args()
    
    config_values = load_config(args.config)
    
    base_output_dir = config_values.get('output_folder_dir', './text_simulation_output_dream_vaccine')
    base_output_dir = os.path.join("./text_simulation", base_output_dir)
    
    num_workers = config_values.get('num_workers', 1)  # Default to 1 for local model
    max_retries_for_sequence = config_values.get('max_retries', 3)
    force_regenerate = config_values.get('force_regenerate', False)
    max_personas = config_values.get('max_personas', None)
    
    # Get start_index and batch_size from args or config
    start_index = args.start_index if args.start_index is not None else config_values.get('start_index', None)
    batch_size = args.batch_size if args.batch_size is not None else config_values.get('batch_size', None)
    
    if args.max_personas:
        max_personas = args.max_personas
        if max_personas == -1:
            max_personas = None
    
    # Get Dream-specific config
    dream_config_dict = config_values.get('dream_config', {})
    if not dream_config_dict:
        # Fallback to top-level config
        dream_config_dict = {
            'model_name': config_values.get('model_name', 'Dream-org/Dream-v0-Instruct-7B'),
            'temperature': config_values.get('temperature', 0.2),
            'max_new_tokens': config_values.get('max_new_tokens', 512),
            'steps': config_values.get('steps', 512),
            'alg': config_values.get('alg', 'entropy'),
            'system_instruction': config_values.get('system_instruction'),
        }
    
    if not args.dataset:
        raise ValueError("--dataset must be specified.")
    
    print(f"Starting Dream 7B vaccine predictions from: {args.config} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {dream_config_dict.get('model_name', 'Dream-org/Dream-v0-Instruct-7B')}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {base_output_dir}")
    print(f"Number of concurrent requests: {num_workers} (local model - typically 1)")
    print(f"Force regenerate: {force_regenerate}")
    if start_index is not None:
        print(f"Start index: {start_index}")
    if batch_size is not None:
        print(f"Batch size: {batch_size}")
    if max_personas:
        print(f"Max personas to process: {max_personas} (legacy parameter)")
    
    asyncio.run(run_vaccine_predictions(
        dataset_path=args.dataset,
        base_output_dir=base_output_dir,
        dream_config_params=dream_config_dict,
        num_workers=num_workers,
        max_retries_for_sequence=max_retries_for_sequence,
        force_regenerate=force_regenerate,
        max_personas=max_personas,
        start_index=start_index,
        batch_size=batch_size
    ))


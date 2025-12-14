"""
Run LLM Simulations for Economic Game Predictions using Dream 7B Model

This script uses demographics + Dictator game + Ultimatum game to predict:
- Trust game sender (QID117)
- Vaccine acceptance (QID291)

Input: Demographics + Dictator answer + Ultimatum sender answer
Output: Predictions for Trust game and Vaccine

It loads a clean dataset with demographics and game answers, then uses Dream 7B 
to predict Trust game and Vaccine responses.

Usage:
    python run_LLM_simulations_dream_vaccine.py --config text_simulation/configs/dream_config.yaml --dataset ./data/demographic_games_dataset.json
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
    create_vaccine_prompt_with_context,
    create_trust_sender_prompt_with_context,
    parse_vaccine_response,
    parse_game_response,
    validate_vaccine_response,
    validate_game_response
)
from datetime import datetime

load_dotenv()


# Updated system instruction for game predictions
GAME_PREDICTION_SYSTEM_INSTRUCTION = """You are an AI assistant that predicts human behavior based on demographic characteristics and past game responses. 
Given demographic information and how a person responded to Dictator and Ultimatum games, predict how they would respond to Trust game and Vaccine questions. 
Respond with only a single number corresponding to the option they would choose."""


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
    output_filename = f"{persona_id}_game_predictions.json"
    return os.path.join(persona_output_folder, output_filename)


# Global storage for predictions (persona_id -> question_type -> prediction_data)
_predictions_storage = {}
import threading
_predictions_lock = threading.Lock()


def save_and_verify_game_callback(
    prompt_id: str, 
    llm_response_data: dict, 
    original_prompt_text: str, 
    **kwargs
) -> bool:
    """
    Saves the Dream LLM response and verifies it contains a valid prediction.
    
    Expected kwargs: base_output_dir, persona_to_entry, question_type
    """
    base_output_dir = kwargs.get("base_output_dir")
    persona_to_entry = kwargs.get("persona_to_entry", {})
    prompt_id_to_question_type = kwargs.get("prompt_id_to_question_type", {})
    question_type = prompt_id_to_question_type.get(prompt_id, "unknown")
    
    if not base_output_dir:
        print(f"Error for {prompt_id}: Missing base_output_dir in verification_callback_args.")
        return False
    
    # Extract persona_id from prompt_id (format: "pid_XXX_questiontype")
    prompt_id_to_persona_id = kwargs.get("prompt_id_to_persona_id", {})
    persona_id = prompt_id_to_persona_id.get(prompt_id)
    if not persona_id:
        # Fallback: try to extract from prompt_id
        import re
        match = re.search(r'(pid_\d+)', prompt_id)
        if match:
            persona_id = match.group(1)
        else:
            print(f"Error for {prompt_id}: Could not extract persona_id.")
            return False
    
    dataset_entry = persona_to_entry.get(persona_id)
    if not dataset_entry:
        print(f"Error for {prompt_id}: No dataset entry found for persona {persona_id}.")
        return False
    response_text = llm_response_data.get("response_text", "")
    
    # Parse response based on question type
    predicted_answer = None
    is_valid = False
    
    if question_type == "vaccine":
        predicted_answer = parse_vaccine_response(response_text)
        is_valid = validate_vaccine_response(response_text)
    elif question_type == "trust_sender":
        predicted_answer = parse_game_response(response_text)
        is_valid = validate_game_response(response_text, min_val=1, max_val=6)
    
    # Store prediction in global storage (thread-safe)
    with _predictions_lock:
        if persona_id not in _predictions_storage:
            _predictions_storage[persona_id] = {}
        _predictions_storage[persona_id][question_type] = {
            "predicted_answer": predicted_answer,
            "response_text": response_text,
            "prompt_text": original_prompt_text,
            "usage_details": llm_response_data.get("usage_details", {}),
            "llm_call_error": llm_response_data.get("error"),
            "is_valid": is_valid
        }
    
    # If the LLM call itself had an error, no point in verifying
    if "error" in llm_response_data and llm_response_data["error"]:
        return False
    
    if not is_valid:
        print(f"Warning: Invalid {question_type} response for {prompt_id}: {response_text[:100]}")
    
    return is_valid


def save_all_predictions(base_output_dir: str, dataset: list):
    """
    Save all collected predictions to JSON files.
    
    This consolidates predictions from all questions for each persona.
    """
    for entry in dataset:
        persona_id = entry["persona_id"]
        if persona_id not in _predictions_storage:
            continue
        
        output_path = get_output_path(base_output_dir, persona_id)
        predictions = _predictions_storage[persona_id]
        
        # Build output JSON with all predictions
        output_data = {
            "persona_id": persona_id,
            "demographic_description": entry.get("demographic_description", ""),
            "predictions": {},
            "actual_answers": {}
        }
        
        # Vaccine prediction
        if "vaccine" in predictions and entry.get("vaccine_question"):
            pred = predictions["vaccine"]
            output_data["predictions"]["vaccine"] = {
                "predicted_answer": pred["predicted_answer"],
                "predicted_answer_text": {
                    1: "I would definitely not take the vaccine",
                    2: "I would probably not take the vaccine",
                    3: "I would probably take the vaccine",
                    4: "I would definitely take the vaccine"
                }.get(pred["predicted_answer"], None),
                "response_text": pred["response_text"],
                "is_valid": pred["is_valid"]
            }
            output_data["actual_answers"]["vaccine"] = {
                "question_id": entry["vaccine_question"]["question_id"],
                "question_text": entry["vaccine_question"]["question_text"],
                "actual_answer": entry["vaccine_question"]["actual_answer"],
                "actual_answer_text": entry["vaccine_question"]["actual_answer_text"]
            }
        
        # Input games (Dictator and Ultimatum) - included for reference
        if entry.get("dictator_game"):
            output_data["input_data"] = output_data.get("input_data", {})
            output_data["input_data"]["dictator_game"] = {
                "question_id": entry["dictator_game"]["question_id"],
                "question_text": entry["dictator_game"]["question_text"],
                "actual_answer": entry["dictator_game"]["actual_answer"],
                "actual_answer_text": entry["dictator_game"]["actual_answer_text"]
            }
        
        if entry.get("ultimatum_game", {}).get("sender"):
            output_data["input_data"] = output_data.get("input_data", {})
            output_data["input_data"]["ultimatum_sender"] = {
                "question_id": entry["ultimatum_game"]["sender"]["question_id"],
                "question_text": entry["ultimatum_game"]["sender"]["question_text"],
                "actual_answer": entry["ultimatum_game"]["sender"]["actual_answer"],
                "actual_answer_text": entry["ultimatum_game"]["sender"]["actual_answer_text"]
            }
        
        # Trust sender prediction
        if "trust_sender" in predictions and entry.get("trust_game", {}).get("sender"):
            pred = predictions["trust_sender"]
            output_data["predictions"]["trust_sender"] = {
                "predicted_answer": pred["predicted_answer"],
                "response_text": pred["response_text"],
                "is_valid": pred["is_valid"]
            }
            output_data["actual_answers"]["trust_sender"] = {
                "question_id": entry["trust_game"]["sender"]["question_id"],
                "question_text": entry["trust_game"]["sender"]["question_text"],
                "actual_answer": entry["trust_game"]["sender"]["actual_answer"],
                "actual_answer_text": entry["trust_game"]["sender"]["actual_answer_text"]
            }
        
        
        # Add metadata
        output_data["model"] = "Dream-7B"
        output_data["usage_details"] = {}
        for pred_data in predictions.values():
            if pred_data.get("usage_details"):
                # Merge usage details (sum tokens if multiple)
                for key, val in pred_data["usage_details"].items():
                    if key.endswith("_token_count"):
                        output_data["usage_details"][key] = output_data["usage_details"].get(key, 0) + val
                    else:
                        output_data["usage_details"][key] = pred_data["usage_details"][key]
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing output file {output_path} for {persona_id}: {e}")


async def run_game_predictions(
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
    Run economic game predictions using Dream 7B model.
    
    Uses: Demographics + Dictator game + Ultimatum game (as input)
    Predicts: Trust game sender (QID117) + Vaccine (QID291)
    
    Args:
        dataset_path: Path to demographic_games_dataset.json
        base_output_dir: Directory to save outputs
        dream_config_params: Dictionary of Dream configuration parameters
        num_workers: Number of concurrent requests (typically 1 for local model)
        max_retries_for_sequence: Max retries for LLM call + verification
        force_regenerate: Whether to regenerate existing outputs
        max_personas: Maximum number of personas to process (None for all)
        start_index: Starting index in sorted list (0-based, for distributed processing)
        batch_size: Number of personas to process from start_index (None for all remaining)
    """
    # Clear global storage
    global _predictions_storage
    _predictions_storage = {}
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} personas from dataset.")
    
    # Use custom system instruction for game predictions
    system_instruction = dream_config_params.get(
        'system_instruction', 
        GAME_PREDICTION_SYSTEM_INSTRUCTION
    )
    
    # Create output directory
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
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
    
    # Prepare all prompts (multiple per persona)
    all_prompts = []
    skipped_due_to_existing_count = 0
    persona_to_entry = {}
    
    for entry in dataset:
        persona_id = entry["persona_id"]
        persona_to_entry[persona_id] = entry
        
        # Check if already processed
        output_path = get_output_path(base_output_dir, persona_id)
        if os.path.exists(output_path) and not force_regenerate:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if existing_data.get("predictions"):
                    skipped_due_to_existing_count += 1
                    continue
            except:
                pass
        
        demographic_description = entry["demographic_description"]
        
        # Get input game answers (Dictator and Ultimatum)
        dictator_answer = None
        if entry.get("dictator_game"):
            dictator_answer = entry["dictator_game"].get("actual_answer")
        
        ultimatum_answer = None
        if entry.get("ultimatum_game", {}).get("sender"):
            ultimatum_answer = entry["ultimatum_game"]["sender"].get("actual_answer")
        
        # Only create prompts if we have the required input data
        has_required_inputs = dictator_answer is not None and ultimatum_answer is not None
        
        if not has_required_inputs:
            continue  # Skip personas without required input data
        
        # Create prompts for prediction targets: Trust game and Vaccine
        if entry.get("trust_game", {}).get("sender"):
            prompt_text = create_trust_sender_prompt_with_context(
                demographic_description,
                dictator_answer=dictator_answer,
                ultimatum_answer=ultimatum_answer
            )
            all_prompts.append({
                "prompt_id": f"{persona_id}_trust_sender",
                "persona_id": persona_id,
                "question_type": "trust_sender",
                "prompt_text": prompt_text
            })
        
        if entry.get("vaccine_question"):
            prompt_text = create_vaccine_prompt_with_context(
                demographic_description,
                dictator_answer=dictator_answer,
                ultimatum_answer=ultimatum_answer
            )
            all_prompts.append({
                "prompt_id": f"{persona_id}_vaccine",
                "persona_id": persona_id,
                "question_type": "vaccine",
                "prompt_text": prompt_text
            })
    
    if skipped_due_to_existing_count > 0:
        print(f"Skipped {skipped_due_to_existing_count} personas as their predictions already exist.")
    
    if not all_prompts:
        print("No new prompts require Dream processing or re-processing.")
        return
    
    print(f"Found {len(all_prompts)} prompts to process ({len(set(p['persona_id'] for p in all_prompts))} unique personas).")
    print(f"Processing with Dream 7B (up to {num_workers} concurrent requests).")
    
    # Create mappings for prompt IDs
    prompt_id_to_question_type = {p["prompt_id"]: p["question_type"] for p in all_prompts}
    prompt_id_to_persona_id = {p["prompt_id"]: p["persona_id"] for p in all_prompts}
    
    # Prepare verification args
    verification_args = {
        "base_output_dir": base_output_dir,
        "persona_to_entry": persona_to_entry,
        "prompt_id_to_question_type": prompt_id_to_question_type,
        "prompt_id_to_persona_id": prompt_id_to_persona_id
    }
    
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
        verification_callback=save_and_verify_game_callback,
        verification_callback_args=verification_args,
        max_context_length=dream_config_params.get('max_context_length', 2048)
    )
    
    # Process prompts in batches by persona to ensure all questions for a persona are processed
    # Convert to format expected by process_prompts_batch_dream
    prompts_for_batch = [(p["prompt_id"], p["prompt_text"]) for p in all_prompts]
    
    # Process all prompts (callback will store predictions in _predictions_storage)
    final_results = await process_prompts_batch_dream(
        prompts_for_batch,
        dream_config,
        desc="Dream 7B Game Predictions"
    )
    
    # Save all collected predictions to files
    save_all_predictions(base_output_dir, dataset)
    
    # Count successes and failures
    successful_count = 0
    failed_count = 0
    
    for prompt_id, result_data in final_results.items():
        if "error" in result_data and result_data["error"]:
            failed_count += 1
        else:
            response_text = result_data.get("response_text", "")
            question_type = prompt_id_to_question_type.get(prompt_id, "unknown")
            if question_type == "vaccine":
                is_valid = validate_vaccine_response(response_text)
            elif question_type == "trust_sender":
                is_valid = validate_game_response(response_text, min_val=1, max_val=6)
            else:
                is_valid = False
            
            if is_valid:
                successful_count += 1
            else:
                failed_count += 1
    
    print(f"\nDream 7B game prediction run complete.")
    print(f"  Successfully processed and verified: {successful_count}")
    print(f"  Failed permanently after all retries: {failed_count}")
    if skipped_due_to_existing_count > 0:
        print(f"  Skipped (already existing): {skipped_due_to_existing_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Dream 7B predictions for Trust game and Vaccine using demographics + Dictator + Ultimatum."
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
    
    print(f"Starting Dream 7B game predictions from: {args.config} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    asyncio.run(run_game_predictions(
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


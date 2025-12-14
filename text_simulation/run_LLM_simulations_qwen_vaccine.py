"""
Run LLM Simulations for Behavioral Predictions using Qwen2.5-7B-Instruct Model

This script uses demographics + psychographics to predict:
- QID34: The Impulse Test
- QID36: Financial Literacy
- QID117: The Trust Game (sender)
- QID250: Risk Aversion
- QID150: Mental Accounting

Input: Demographics + Predictor variables (QID26, QID31, QID239, QID35)
Output: Predictions for all 5 target variables

It loads a clean dataset with demographics and predictor/predicted variables, then uses Qwen2.5-7B-Instruct 
to predict behavioral responses.

Usage:
    python run_LLM_simulations_qwen_vaccine.py --config text_simulation/configs/qwen_config.yaml --dataset ./data/demographic_games_dataset.csv
"""

import os
import json
import csv
import argparse
import yaml
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv
from llm_helper_qwen import QwenLLMConfig, process_prompts_batch_qwen
from vaccine_prediction_helpers import (
    get_trust_sender_question_text,
    parse_game_response,
    validate_game_response
)
from datetime import datetime

load_dotenv()


# Updated system instruction for behavioral predictions
BEHAVIORAL_PREDICTION_SYSTEM_INSTRUCTION = """You are an AI assistant that predicts human behavior based on demographic characteristics and psychographic traits. 
Given demographic information and psychographic measures (Need for Cognition, Spendthrift/Tightwad, Maximization, Minimalism), predict how a person would respond to behavioral questions. 
Respond with only a single number corresponding to the option they would choose."""


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data


def load_dataset(dataset_path: str) -> list:
    """
    Load the demographic-vaccine dataset from CSV or JSON.
    
    Returns a list of dictionaries compatible with the existing code structure.
    """
    if dataset_path.endswith('.csv'):
        # Load CSV
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Check available columns
            available_columns = reader.fieldnames
            print(f"CSV columns found: {', '.join(available_columns)}")
            
            vaccine_count = 0
            for row in reader:
                # Convert to expected format
                entry = {
                    "persona_id": row['persona_id'],
                    "demographic_description": row['demographic_description'],
                }
                
                # Parse predictor variables (QID26, QID31, QID239, QID35)
                for qid in ['qid26', 'qid31', 'qid239', 'qid35']:
                    ans = row.get(f'{qid}_answer', '').strip()
                    if ans:
                        try:
                            entry[f"{qid}_question"] = {"actual_answer": int(ans)}
                        except ValueError:
                            pass
                
                # Parse predicted variables (QID34, QID36, QID117, QID250, QID150)
                # QID117 is trust_game
                trust_ans = row.get('qid117_answer', '').strip()
                if trust_ans:
                    try:
                        entry["trust_game"] = {"sender": {"actual_answer": int(trust_ans)}}
                    except ValueError:
                        pass
                
                for qid in ['qid34', 'qid36', 'qid250', 'qid150']:
                    ans = row.get(f'{qid}_answer', '').strip()
                    if ans:
                        try:
                            entry[f"{qid}_question"] = {"actual_answer": int(ans)}
                        except ValueError:
                            pass
                
                dataset.append(entry)
            
            # Count personas with predictor and predicted variables
            predictor_count = sum(1 for e in dataset if any(e.get(f'qid{qid}_question') for qid in [26, 31, 239, 35]))
            predicted_count = sum(1 for e in dataset if any([
                e.get('trust_game', {}).get('sender'),
                e.get('qid34_question'),
                e.get('qid36_question'),
                e.get('qid250_question'),
                e.get('qid150_question')
            ]))
            print(f"Loaded {len(dataset)} personas. {predictor_count} have predictor variables, {predicted_count} have predicted variables.")
        return dataset
    else:
        # Load JSON (backward compatibility)
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset


def get_output_path(base_output_dir: str, persona_id: str, csv_mode: bool = False) -> str:
    """
    Get output file path for a persona prediction.
    If csv_mode=True, returns path to consolidated CSV file instead.
    """
    if csv_mode:
        # Return path to consolidated CSV file
        return os.path.join(base_output_dir, "game_predictions.csv")
    else:
        # Return path to individual JSON file (backward compatibility)
        persona_output_folder = os.path.join(base_output_dir, persona_id)
        os.makedirs(persona_output_folder, exist_ok=True)
        output_filename = f"{persona_id}_game_predictions.json"
        return os.path.join(persona_output_folder, output_filename)


# Global storage for predictions (persona_id -> question_type -> prediction_data)
_predictions_storage = {}
import threading
_predictions_lock = threading.Lock()


def create_generic_question_prompt_with_context(
    demographic_description: str,
    question_text: str,
    predictor_answers: dict = None
) -> str:
    """
    Create a generic prompt for any question prediction using demographics + predictor variables.
    
    Args:
        demographic_description: Natural language description of demographics
        question_text: The question text to predict
        predictor_answers: Dict with predictor variable answers (qid26, qid31, qid239, qid35)
        
    Returns:
        Complete prompt string
    """
    context_parts = [f"## Demographic Profile:\n{demographic_description}"]
    
    # Add predictor variables if provided
    if predictor_answers:
        predictor_labels = {
            'qid26': 'Need for Cognition',
            'qid31': 'Spendthrift/Tightwad',
            'qid239': 'Maximization',
            'qid35': 'Minimalism'
        }
        predictor_parts = []
        for qid, label in predictor_labels.items():
            if qid in predictor_answers and predictor_answers[qid] is not None:
                predictor_parts.append(f"- {label}: {predictor_answers[qid]}")
        
        if predictor_parts:
            context_parts.append(f"## Psychographic Traits:\n" + "\n".join(predictor_parts))
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""{context}

---

## Question to Predict:
{question_text}

## Instructions:
Based on the demographic profile and psychographic traits above, predict how this person would respond.
Respond with only a single number corresponding to the option they would choose."""
    
    return prompt


def save_and_verify_game_callback(
    prompt_id: str, 
    llm_response_data: dict, 
    original_prompt_text: str, 
    **kwargs
) -> bool:
    """
    Saves the Qwen LLM response and verifies it contains a valid prediction.
    
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
    
    # All questions use numeric responses, validate based on expected range
    if question_type == "qid117":  # Trust game (1-6)
        predicted_answer = parse_game_response(response_text)
        is_valid = validate_game_response(response_text, min_val=1, max_val=6)
    elif question_type in ["qid34", "qid36", "qid250", "qid150"]:
        # Generic numeric parsing for other questions
        predicted_answer = parse_game_response(response_text)
        # Default validation: 1-10 range (adjust if needed based on actual question options)
        is_valid = validate_game_response(response_text, min_val=1, max_val=10)
    
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
        print(f"Warning: Invalid {question_type} response for {prompt_id}: {response_text[:200]}")
        # Still store the prediction even if invalid, so we can see what the model generated
        if predicted_answer is None:
            print(f"  → Could not parse number from response. Full response: {response_text}")
    
    return is_valid


def save_all_predictions(base_output_dir: str, dataset: list, csv_mode: bool = True):
    """
    Save all collected predictions to a single CSV file (or JSON files for backward compatibility).
    
    CSV format: persona_id, predictor variables, predicted variables with actual and predicted values
    """
    os.makedirs(base_output_dir, exist_ok=True)
    
    if csv_mode:
        # Save as single CSV file
        csv_path = get_output_path(base_output_dir, "", csv_mode=True)
        
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = [
                'persona_id',
                # Predictor variables
                'qid26_input', 'qid31_input', 'qid239_input', 'qid35_input',
                # Predicted variables
                'qid34_predicted', 'qid34_actual', 'qid34_valid',
                'qid36_predicted', 'qid36_actual', 'qid36_valid',
                'qid117_predicted', 'qid117_actual', 'qid117_valid',
                'qid250_predicted', 'qid250_actual', 'qid250_valid',
                'qid150_predicted', 'qid150_actual', 'qid150_valid'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in dataset:
                persona_id = entry["persona_id"]
                if persona_id not in _predictions_storage:
                    continue
                
                predictions = _predictions_storage[persona_id]
                
                row = {
                    'persona_id': persona_id,
                    # Predictor variables
                    'qid26_input': entry.get('qid26_question', {}).get('actual_answer', ''),
                    'qid31_input': entry.get('qid31_question', {}).get('actual_answer', ''),
                    'qid239_input': entry.get('qid239_question', {}).get('actual_answer', ''),
                    'qid35_input': entry.get('qid35_question', {}).get('actual_answer', ''),
                    # Predicted variables
                    'qid34_predicted': predictions.get('qid34', {}).get('predicted_answer', ''),
                    'qid34_actual': entry.get('qid34_question', {}).get('actual_answer', ''),
                    'qid34_valid': predictions.get('qid34', {}).get('is_valid', False),
                    'qid36_predicted': predictions.get('qid36', {}).get('predicted_answer', ''),
                    'qid36_actual': entry.get('qid36_question', {}).get('actual_answer', ''),
                    'qid36_valid': predictions.get('qid36', {}).get('is_valid', False),
                    'qid117_predicted': predictions.get('qid117', {}).get('predicted_answer', ''),
                    'qid117_actual': entry.get('trust_game', {}).get('sender', {}).get('actual_answer', ''),
                    'qid117_valid': predictions.get('qid117', {}).get('is_valid', False),
                    'qid250_predicted': predictions.get('qid250', {}).get('predicted_answer', ''),
                    'qid250_actual': entry.get('qid250_question', {}).get('actual_answer', ''),
                    'qid250_valid': predictions.get('qid250', {}).get('is_valid', False),
                    'qid150_predicted': predictions.get('qid150', {}).get('predicted_answer', ''),
                    'qid150_actual': entry.get('qid150_question', {}).get('actual_answer', ''),
                    'qid150_valid': predictions.get('qid150', {}).get('is_valid', False),
                }
                writer.writerow(row)
        
        print(f"Predictions saved to {csv_path} ({len([e for e in dataset if e['persona_id'] in _predictions_storage])} personas)")
        return
    
    # Backward compatibility: save individual JSON files
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
        
        # Add all predicted variables
        for qid in ['qid34', 'qid36', 'qid117', 'qid250', 'qid150']:
            if qid in predictions:
                pred = predictions[qid]
                output_data["predictions"][qid] = {
                    "predicted_answer": pred["predicted_answer"],
                    "response_text": pred["response_text"],
                    "is_valid": pred["is_valid"]
                }
                # Add actual answer if available
                if qid == 'qid117':
                    if entry.get("trust_game", {}).get("sender"):
                        output_data["actual_answers"][qid] = {
                            "actual_answer": entry["trust_game"]["sender"]["actual_answer"]
                        }
                elif entry.get(f"{qid}_question"):
                    output_data["actual_answers"][qid] = {
                        "actual_answer": entry[f"{qid}_question"]["actual_answer"]
                    }
        
        # Input predictor variables - included for reference
        if any(entry.get(f"{qid}_question") for qid in ['qid26', 'qid31', 'qid239', 'qid35']):
            output_data["input_data"] = output_data.get("input_data", {})
            for qid in ['qid26', 'qid31', 'qid239', 'qid35']:
                if entry.get(f"{qid}_question"):
                    output_data["input_data"][qid] = {
                        "actual_answer": entry[f"{qid}_question"]["actual_answer"]
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
        output_data["model"] = "Qwen2.5-7B-Instruct"
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
    qwen_config_params: dict,
    num_workers: int,
    max_retries_for_sequence: int,
    force_regenerate: bool,
    max_personas=None,
    start_index=None,
    batch_size=None
):
    """
    Run behavioral predictions using Qwen2.5-7B-Instruct model.
    
    Uses: Demographics + Psychographics (QID26, QID31, QID239, QID35) as input
    Predicts: QID34 (Impulse Test), QID36 (Financial Literacy), QID117 (Trust Game), QID250 (Risk Aversion), QID150 (Mental Accounting)
    
    Args:
        dataset_path: Path to demographic_games_dataset.csv
        base_output_dir: Directory to save outputs
        qwen_config_params: Dictionary of Qwen configuration parameters
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
    
    # Use custom system instruction for behavioral predictions
    system_instruction = qwen_config_params.get(
        'system_instruction', 
        BEHAVIORAL_PREDICTION_SYSTEM_INSTRUCTION
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
        
        # Get predictor variable answers (QID26, QID31, QID239, QID35)
        predictor_answers = {}
        for qid in ['qid26', 'qid31', 'qid239', 'qid35']:
            if entry.get(f"{qid}_question"):
                predictor_answers[qid] = entry[f"{qid}_question"].get("actual_answer")
        
        # Only create prompts if we have at least some predictor data
        has_predictors = any(predictor_answers.values())
        
        if not has_predictors:
            continue  # Skip personas without predictor data
        
        # Create prompts for all predicted variables
        # QID117: Trust Game
        if entry.get("trust_game", {}).get("sender") or True:  # Always predict
            prompt_text = create_generic_question_prompt_with_context(
                demographic_description,
                get_trust_sender_question_text().replace('Based on the demographic profile above', 'Based on the information above'),
                predictor_answers
            )
            all_prompts.append({
                "prompt_id": f"{persona_id}_qid117",
                "persona_id": persona_id,
                "question_type": "qid117",
                "prompt_text": prompt_text
            })
        
        # QID34: Impulse Test
        # Note: Question text will need to be provided - placeholder for now
        if entry.get("qid34_question") or True:  # Always predict
            question_text = "QID34: The Impulse Test\n[Question text needed - please add from question catalog]"
            prompt_text = create_generic_question_prompt_with_context(
                demographic_description,
                question_text,
                predictor_answers
            )
            all_prompts.append({
                "prompt_id": f"{persona_id}_qid34",
                "persona_id": persona_id,
                "question_type": "qid34",
                "prompt_text": prompt_text
            })
        
        # QID36: Financial Literacy
        if entry.get("qid36_question") or True:  # Always predict
            question_text = "QID36: Financial Literacy\n[Question text needed - please add from question catalog]"
            prompt_text = create_generic_question_prompt_with_context(
                demographic_description,
                question_text,
                predictor_answers
            )
            all_prompts.append({
                "prompt_id": f"{persona_id}_qid36",
                "persona_id": persona_id,
                "question_type": "qid36",
                "prompt_text": prompt_text
            })
        
        # QID250: Risk Aversion
        if entry.get("qid250_question") or True:  # Always predict
            question_text = "QID250: Risk Aversion\n[Question text needed - please add from question catalog]"
            prompt_text = create_generic_question_prompt_with_context(
                demographic_description,
                question_text,
                predictor_answers
            )
            all_prompts.append({
                "prompt_id": f"{persona_id}_qid250",
                "persona_id": persona_id,
                "question_type": "qid250",
                "prompt_text": prompt_text
            })
        
        # QID150: Mental Accounting
        if entry.get("qid150_question") or True:  # Always predict
            question_text = "QID150: Mental Accounting\n[Question text needed - please add from question catalog]"
            prompt_text = create_generic_question_prompt_with_context(
                demographic_description,
                question_text,
                predictor_answers
            )
            all_prompts.append({
                "prompt_id": f"{persona_id}_qid150",
                "persona_id": persona_id,
                "question_type": "qid150",
                "prompt_text": prompt_text
            })
    
    if skipped_due_to_existing_count > 0:
        print(f"Skipped {skipped_due_to_existing_count} personas as their predictions already exist.")
    
    if not all_prompts:
        print("No new prompts require Qwen processing or re-processing.")
        return
    
    # Count prompts by type
    qid_counts = {}
    for qid in ['qid34', 'qid36', 'qid117', 'qid250', 'qid150']:
        qid_counts[qid] = sum(1 for p in all_prompts if p["question_type"] == qid)
    
    print(f"Found {len(all_prompts)} prompts to process ({len(set(p['persona_id'] for p in all_prompts))} unique personas).")
    for qid, count in qid_counts.items():
        print(f"  - {qid} prompts: {count}")
    print(f"Processing with Qwen2.5-7B-Instruct (up to {num_workers} concurrent requests).")
    
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
    
    # Create Qwen config
    qwen_config = QwenLLMConfig(
        model_name=qwen_config_params.get('model_name', 'Qwen/Qwen2.5-7B-Instruct'),
        temperature=qwen_config_params.get('temperature', 0.2),
        max_new_tokens=qwen_config_params.get('max_new_tokens', 512),
        top_p=qwen_config_params.get('top_p', 0.95),
        top_k=qwen_config_params.get('top_k', None),
        system_instruction=system_instruction,
        max_retries=max_retries_for_sequence,
        max_concurrent_requests=num_workers,
        device=qwen_config_params.get('device', 'cuda'),
        verification_callback=save_and_verify_game_callback,
        verification_callback_args=verification_args,
        max_context_length=qwen_config_params.get('max_context_length', 32768)
    )
    
    # Process prompts in batches by persona to ensure all questions for a persona are processed
    # Convert to format expected by process_prompts_batch_qwen
    prompts_for_batch = [(p["prompt_id"], p["prompt_text"]) for p in all_prompts]
    
    # Process all prompts (callback will store predictions in _predictions_storage)
    final_results = await process_prompts_batch_qwen(
        prompts_for_batch,
        qwen_config,
        desc="Qwen2.5-7B-Instruct Game Predictions"
    )
    
    # Save all collected predictions to CSV file (fast, single file write)
    save_all_predictions(base_output_dir, dataset, csv_mode=True)
    
    # Count successes and failures
    successful_count = 0
    failed_count = 0
    
    for prompt_id, result_data in final_results.items():
        if "error" in result_data and result_data["error"]:
            failed_count += 1
        else:
            response_text = result_data.get("response_text", "")
            question_type = prompt_id_to_question_type.get(prompt_id, "unknown")
            if question_type == "qid117":
                is_valid = validate_game_response(response_text, min_val=1, max_val=6)
            elif question_type in ["qid34", "qid36", "qid250", "qid150"]:
                is_valid = validate_game_response(response_text, min_val=1, max_val=10)
            else:
                is_valid = False
            
            if is_valid:
                successful_count += 1
            else:
                failed_count += 1
    
    # Count predictions by type
    prediction_counts = {}
    valid_counts = {}
    for qid in ['qid34', 'qid36', 'qid117', 'qid250', 'qid150']:
        prediction_counts[qid] = sum(1 for e in dataset if e.get('persona_id') in _predictions_storage and qid in _predictions_storage[e['persona_id']])
        valid_counts[qid] = sum(1 for e in dataset if e.get('persona_id') in _predictions_storage and _predictions_storage[e['persona_id']].get(qid, {}).get('is_valid', False))
    
    print(f"\nQwen2.5-7B-Instruct behavioral prediction run complete.")
    print(f"  Successfully processed and verified: {successful_count}")
    print(f"  Failed permanently after all retries: {failed_count}")
    print(f"\n  Prediction Summary:")
    for qid in ['qid34', 'qid36', 'qid117', 'qid250', 'qid150']:
        label = {
            'qid34': 'Impulse Test',
            'qid36': 'Financial Literacy',
            'qid117': 'Trust Game',
            'qid250': 'Risk Aversion',
            'qid150': 'Mental Accounting'
        }.get(qid, qid)
        print(f"    {qid} ({label}): {prediction_counts[qid]} predictions ({valid_counts[qid]} valid)")
    if skipped_due_to_existing_count > 0:
        print(f"  Skipped (already existing): {skipped_due_to_existing_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-7B-Instruct predictions for behavioral variables using demographics + psychographics."
    )
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--dataset",
        default="./data/demographic_games_dataset.csv",
        help="Path to demographic_games_dataset.csv"
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
    
    base_output_dir = config_values.get('output_folder_dir', './text_simulation_output_qwen_vaccine')
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
    
    # Get Qwen-specific config
    qwen_config_dict = config_values.get('qwen_config', {})
    if not qwen_config_dict:
        # Fallback to top-level config
        qwen_config_dict = {
            'model_name': config_values.get('model_name', 'Qwen/Qwen2.5-7B-Instruct'),
            'temperature': config_values.get('temperature', 0.2),
            'max_new_tokens': config_values.get('max_new_tokens', 512),
            'system_instruction': config_values.get('system_instruction'),
        }
    
    if not args.dataset:
        raise ValueError("--dataset must be specified.")
    
    print(f"Starting Qwen2.5-7B-Instruct game predictions from: {args.config} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {qwen_config_dict.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')}")
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
        qwen_config_params=qwen_config_dict,
        num_workers=num_workers,
        max_retries_for_sequence=max_retries_for_sequence,
        force_regenerate=force_regenerate,
        max_personas=max_personas,
        start_index=start_index,
        batch_size=batch_size
    ))


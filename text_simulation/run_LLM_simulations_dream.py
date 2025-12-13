"""
Run LLM Simulations using Dream 7B Model

This script is adapted from run_LLM_simulations.py to use Dream 7B instead of
OpenAI/Gemini API models. Dream 7B is a local diffusion language model.

Usage:
    python run_LLM_simulations_dream.py --config text_simulation/configs/dream_config.yaml --max_personas 5
"""

import os
import json
import argparse
import re
import yaml
from tqdm import tqdm
import asyncio
from dotenv import load_dotenv
from postprocess_responses import postprocess_simulation_outputs_with_pid
from llm_helper_dream import DreamLLMConfig, process_prompts_batch_dream
from datetime import datetime

load_dotenv()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data


def get_output_path(base_output_dir, persona_id, question_id):
    """Get output file path for a persona-question pair."""
    persona_output_folder = os.path.join(base_output_dir, persona_id)
    os.makedirs(persona_output_folder, exist_ok=True)
    output_filename = f"{question_id}_response.json"
    return os.path.join(persona_output_folder, output_filename)


def save_and_verify_callback(prompt_id: str, llm_response_data: dict, original_prompt_text: str, **kwargs) -> bool:
    """
    Saves the Dream LLM response and then verifies it.
    Expected kwargs: base_output_dir, question_json_base_dir, output_updated_questions_dir_for_verify
    """
    base_output_dir = kwargs.get("base_output_dir")
    question_json_base_dir = kwargs.get("question_json_base_dir")
    output_updated_questions_dir_for_verify = kwargs.get("output_updated_questions_dir_for_verify")

    if not all([base_output_dir, question_json_base_dir, output_updated_questions_dir_for_verify]):
        print(f"Error for {prompt_id}: Missing critical path arguments in verification_callback_args.")
        return False

    persona_id = prompt_id
    question_id = persona_id

    output_path = get_output_path(base_output_dir, persona_id, question_id)

    # Save Dream response
    output_json_data = {
        "persona_id": persona_id,
        "question_id": question_id,
        "prompt_text": original_prompt_text,
        "response_text": llm_response_data.get("response_text", ""),
        "usage_details": llm_response_data.get("usage_details", {}),
        "llm_call_error": llm_response_data.get("error"),
        "model": "Dream-7B"  # Mark as Dream model
    }
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json_data, f, indent=2)
    except Exception as e:
        print(f"Error writing output file {output_path} for {prompt_id}: {e}")
        return False

    # If the LLM call itself had an error, no point in verifying
    if "error" in llm_response_data and llm_response_data["error"]:
        return False

    # Verify the output
    try:
        is_verified = postprocess_simulation_outputs_with_pid(
            persona_id,
            base_output_dir,
            question_json_base_dir,
            output_updated_questions_dir_for_verify
        )
        return is_verified
    except Exception as e:
        print(f"Error during verification call for persona {prompt_id}: {e}")
        return False


async def run_simulations(
    prompts_root_dir,
    base_output_dir,
    dream_config_params,
    num_workers,
    max_retries_for_sequence,
    force_regenerate,
    max_personas=None
):
    """
    Run Dream 7B simulations for all prompts.
    
    Args:
        prompts_root_dir: Directory containing prompt files
        base_output_dir: Directory to save outputs
        dream_config_params: Dictionary of Dream configuration parameters
        num_workers: Number of concurrent requests (typically 1 for local model)
        max_retries_for_sequence: Max retries for LLM call + verification
        force_regenerate: Whether to regenerate existing outputs
        max_personas: Maximum number of personas to process (None for all)
    """
    question_json_base_dir_for_verify = "./data/mega_persona_json/answer_blocks"
    output_updated_questions_dir_for_verify = os.path.join(base_output_dir, "answer_blocks_llm_imputed")

    # Prepare arguments for the verification callback
    verification_args = {
        "base_output_dir": base_output_dir,
        "question_json_base_dir": question_json_base_dir_for_verify,
        "output_updated_questions_dir_for_verify": output_updated_questions_dir_for_verify
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
        system_instruction=dream_config_params.get('system_instruction'),
        max_retries=max_retries_for_sequence,
        max_concurrent_requests=num_workers,
        device=dream_config_params.get('device', 'cuda'),
        verification_callback=save_and_verify_callback,
        verification_callback_args=verification_args,
        max_context_length=dream_config_params.get('max_context_length', 2048)
    )

    # Find all prompt files
    all_prompt_files_info = []
    try:
        prompt_files_fs = sorted([f for f in os.listdir(prompts_root_dir) if f.endswith('_prompt.txt')])
    except FileNotFoundError:
        print(f"Error: Prompts root directory not found: {prompts_root_dir}")
        return

    if max_personas is not None and max_personas > 0:
        prompt_files_fs = prompt_files_fs[:max_personas]
        print(f"Limiting processing to {max_personas} prompt files")

    for prompt_filename in prompt_files_fs:
        persona_match = re.search(r'(pid_\d+)', prompt_filename)
        if persona_match:
            persona_id = persona_match.group(1)
            full_prompt_path = os.path.join(prompts_root_dir, prompt_filename)
            all_prompt_files_info.append({'persona_id': persona_id, 'file_path': full_prompt_path})

    if not all_prompt_files_info:
        print(f"No prompt files (ending with '_prompt.txt') found in {prompts_root_dir}")
        return

    # Create output directories
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    if not os.path.exists(output_updated_questions_dir_for_verify):
        os.makedirs(output_updated_questions_dir_for_verify)

    # Check for existing outputs
    prompts_to_process = []
    skipped_due_to_existing_verified_count = 0

    for info in all_prompt_files_info:
        p_id, f_path = info['persona_id'], info['file_path']
        output_json_path = get_output_path(base_output_dir, p_id, p_id)
        
        if os.path.exists(output_json_path) and not force_regenerate:
            # Check if existing output is valid
            if postprocess_simulation_outputs_with_pid(
                p_id, base_output_dir, question_json_base_dir_for_verify, output_updated_questions_dir_for_verify
            ):
                skipped_due_to_existing_verified_count += 1
                continue

        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            prompts_to_process.append((p_id, prompt_content))
        except Exception as e:
            print(f"Error reading prompt file {f_path} for {p_id}: {e}")

    if skipped_due_to_existing_verified_count > 0:
        print(f"Skipped {skipped_due_to_existing_verified_count} files as their output already exists and is verified.")

    if not prompts_to_process:
        print("No new files require Dream processing or re-processing.")
        if skipped_due_to_existing_verified_count == 0:
            print("No prompt files found or processed in this run.")
        return

    print(f"Found {len(prompts_to_process)} prompts to process with Dream 7B (up to {dream_config.max_concurrent_requests} concurrent requests).")
    print(f"Note: Dream is a local model. Processing may be slower than API-based models.")

    # Process prompts with Dream
    final_results = await process_prompts_batch_dream(
        prompts_to_process,
        dream_config,
        desc="Dream 7B calls & Verification"
    )

    # Count successes and failures
    successful_final_count = 0
    failed_final_count = 0

    for prompt_id, result_data in final_results.items():
        if "error" in result_data and result_data["error"]:
            failed_final_count += 1
            print(f"FINAL FAILURE for {prompt_id}: {result_data['error']}")
            error_file_path = get_output_path(base_output_dir, prompt_id, prompt_id).replace(".json", "_final_error.txt")
            with open(error_file_path, 'w') as ef:
                ef.write(f"Final processing failed for {prompt_id}. Details: {json.dumps(result_data, indent=2)}")
        else:
            successful_final_count += 1

    print(f"\nDream 7B processing run complete.")
    print(f"  Successfully processed and verified: {successful_final_count}")
    print(f"  Failed permanently after all retries: {failed_final_count}")
    if skipped_due_to_existing_verified_count > 0:
        print(f"  Skipped (already existing and verified): {skipped_due_to_existing_verified_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dream 7B simulations with integrated verification and retries.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--max_personas", type=int, help="Maximum number of personas to process")
    args = parser.parse_args()

    config_values = load_config(args.config)

    prompts_root_dir = config_values.get('input_folder_dir', './text_simulation_input')
    base_output_dir = config_values.get('output_folder_dir', './text_simulation_output_dream')
    prompts_root_dir = os.path.join("./text_simulation", prompts_root_dir)
    base_output_dir = os.path.join("./text_simulation", base_output_dir)
    
    num_workers = config_values.get('num_workers', 1)  # Default to 1 for local model
    max_retries_for_sequence = config_values.get('max_retries', 3)
    force_regenerate = config_values.get('force_regenerate', False)
    max_personas = config_values.get('max_personas', None)
    
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

    if not prompts_root_dir or not base_output_dir:
        raise ValueError("prompts_root_dir and base_output_dir must be specified.")

    print(f"Starting Dream 7B simulations from: {args.config} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {dream_config_dict.get('model_name', 'Dream-org/Dream-v0-Instruct-7B')}")
    print(f"Input prompts directory: {prompts_root_dir}")
    print(f"Base output directory: {base_output_dir}")
    print(f"Number of concurrent requests: {num_workers} (local model - typically 1)")
    print(f"Force regenerate: {force_regenerate}")
    if max_personas:
        print(f"Max personas to process: {max_personas}")

    asyncio.run(run_simulations(
        prompts_root_dir=prompts_root_dir,
        base_output_dir=base_output_dir,
        dream_config_params=dream_config_dict,
        num_workers=num_workers,
        max_retries_for_sequence=max_retries_for_sequence,
        force_regenerate=force_regenerate,
        max_personas=max_personas
    ))


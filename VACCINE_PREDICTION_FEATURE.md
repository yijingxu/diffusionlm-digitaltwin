# Vaccine Prediction Feature

## Overview

This feature implements a simplified demographic-only approach for predicting vaccine acceptance likelihood using LLMs. The system extracts only demographic data from personas and predicts their response to the vaccine question (QID291) using Dream 7B model.

## Architecture

The implementation follows a two-step approach:

1. **Data Extraction**: Extract demographics and vaccine answers into a clean dataset
2. **Prediction**: Use demographic descriptions to predict vaccine acceptance (1-4 scale)

## New Files

### 1. `text_simulation/extract_demographic_vaccine_dataset.py`

**Purpose**: Extract demographic data and vaccine answers from persona JSON files into a clean, structured dataset.

**Features**:
- Extracts 14 demographic questions (QID11-QID24) from Wave 1
- Extracts QID291 (vaccine question) answer
- Formats demographics as natural language descriptions
- Creates a clean JSON dataset for downstream processing

**Usage**:
```bash
python text_simulation/extract_demographic_vaccine_dataset.py \
  --input_dir ./data/mega_persona_json/mega_persona \
  --output_file ./data/demographic_vaccine_dataset.json
```

**Output Format**:
- JSON file with entries containing:
  - `persona_id`: Persona identifier
  - `demographics`: Raw demographic Q&A data
  - `demographic_description`: Natural language description
  - `vaccine_question`: QID291 question and actual answer

### 2. `text_simulation/vaccine_prediction_helpers.py`

**Purpose**: Helper functions for vaccine prediction workflow.

**Functions**:
- `get_vaccine_question_text()`: Returns the vaccine question prompt text
- `create_vaccine_prompt(demographic_description)`: Creates complete prompt with demographics and question
- `parse_vaccine_response(response_text)`: Extracts 1-4 prediction from LLM response
- `validate_vaccine_response(response_text)`: Validates prediction is in valid range

### 3. `text_simulation/run_LLM_simulations_dream_vaccine.py`

**Purpose**: Main script for running vaccine predictions using Dream 7B model.

**Features**:
- Loads clean demographic-vaccine dataset
- Creates prompts using demographic descriptions
- Uses Dream 7B to predict vaccine likelihood (1-4)
- Validates and saves predictions with both predicted and actual answers

**Usage**:
```bash
python text_simulation/run_LLM_simulations_dream_vaccine.py \
  --config text_simulation/configs/dream_config.yaml \
  --dataset ./data/demographic_vaccine_dataset.json
```

**Output**:
- Saves predictions to: `{base_output_dir}/{persona_id}/{persona_id}_vaccine_prediction.json`
- Each file contains:
  - Demographic description
  - Predicted likelihood (1-4)
  - Actual answer (for evaluation)
  - Prompt and response details

## Workflow

### Step 1: Extract Dataset
```bash
python text_simulation/extract_demographic_vaccine_dataset.py \
  --input_dir ./data/mega_persona_json/mega_persona \
  --output_file ./data/demographic_vaccine_dataset.json
```

### Step 2: Run Predictions
```bash
python text_simulation/run_LLM_simulations_dream_vaccine.py \
  --config text_simulation/configs/dream_config.yaml \
  --dataset ./data/demographic_vaccine_dataset.json \
  --batch_size 10  # Optional: process in batches
```

## Prompt Format

The prompt uses a simplified format compared to the full persona approach:

```
## Demographic Profile:
This person is [age], lives in [region], identifies as [gender], 
has [education], is [race], [citizenship status], [marital status], 
follows [religion], attends religious services [frequency], 
identifies as [political affiliation], has [income], describes 
political views as [views], lives in a household of [size], 
and is currently [employment status].

---

## Question:
[Vaccine question text with 4 options]

## Instructions:
Based on the demographic profile above, predict on a scale of 1-4 
how likely this person is to accept the vaccine.
Respond with only a single number (1, 2, 3, or 4).
```

## System Instruction

The system instruction for Dream 7B is customized for vaccine prediction:

```
You are an AI assistant that predicts human behavior based on 
demographic characteristics. Given demographic information about 
a person, predict how they would respond to a specific question 
about vaccine acceptance. Respond with only a single number from 1-4, 
where:
1 = I would definitely not take the vaccine
2 = I would probably not take the vaccine
3 = I would probably take the vaccine
4 = I would definitely take the vaccine
```

## Evaluation

Each prediction output includes:
- `predicted_likelihood`: Model's prediction (1-4)
- `actual_answer`: Ground truth from survey data (1-4)
- `predicted_likelihood_text`: Human-readable prediction text
- `actual_answer_text`: Human-readable actual answer text

This enables easy comparison and evaluation of prediction accuracy.

## Differences from Full Persona Approach

| Aspect | Full Persona | Demographic-Only |
|--------|--------------|------------------|
| Input Data | All survey responses | Only demographics (14 questions) |
| Prompt Complexity | Full persona profile | Concise demographic description |
| Output Format | JSON with multiple questions | Single number (1-4) |
| Verification | Complex multi-question validation | Simple numerical validation |
| Use Case | General prediction tasks | Specific demographic-based predictions |

## Benefits

1. **Simplified**: Focuses on demographic factors only
2. **Faster**: Shorter prompts, faster inference
3. **Clearer**: Easier to interpret demographic effects
4. **Evaluation-Ready**: Includes ground truth for comparison
5. **Clean Dataset**: Separated data extraction from prediction

## Future Enhancements

- Support for other LLM models (OpenAI, Gemini, LLaDA)
- Batch processing optimizations
- Evaluation metrics and analysis scripts
- Visualization of demographic effects on predictions


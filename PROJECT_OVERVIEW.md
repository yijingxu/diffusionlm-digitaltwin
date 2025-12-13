# Digital-Twin-Simulation Project Overview

## What is This Project?

**Digital-Twin-Simulation** is a system that creates **virtual representations (digital twins)** of real people based on their survey responses. The system uses Large Language Models (LLMs) to simulate how these individuals would respond to new survey questions, maintaining consistency with their original personality profiles and past responses.

### Key Concept
- **Input**: Survey responses from 2,000+ people across 4 waves of questions (500+ questions total)
- **Process**: Convert responses into "persona profiles" (text summaries of each person)
- **Output**: Use LLMs to predict how each person would answer new questions they haven't seen before

### Dataset: Twin-2K-500
- **2,058 personas** (people) with complete survey data
- **4 waves** of survey questions covering:
  - Demographics
  - Personality traits
  - Cognitive tests
  - Economic preferences
  - Decision-making scenarios
  - And more...

---

## Project Structure & Scripts

### 📥 **Data Download**

#### `download_dataset.py`
**Purpose**: Downloads the Twin-2K-500 dataset from Hugging Face

**What it does**:
1. Downloads wave split data (persona JSON files and answer blocks)
2. Downloads full persona data (persona summaries)
3. Downloads raw CSV files for all 4 waves
4. Organizes data into `data/` directory structure

**Output**:
- `data/mega_persona_json/mega_persona/` - JSON persona files
- `data/mega_persona_json/answer_blocks/` - Answer block JSON files
- `data/mega_persona_summary_text/` - Text summaries of personas
- `data/wave_csv/` - Raw CSV files for each wave

---

### 🤖 **Text Simulation Pipeline** (`text_simulation/`)

This is the core simulation system that uses LLMs to generate responses.

#### `batch_convert_personas.py`
**Purpose**: Converts persona JSON files to text format for LLM processing

**What it does**:
- Reads persona JSON files from `data/mega_persona_json/mega_persona/`
- Converts them to readable text format
- Saves to `text_simulation/text_personas/`

**Usage**:
```bash
python text_simulation/batch_convert_personas.py \
    --persona_json_dir data/mega_persona_json/mega_persona \
    --output_text_dir text_simulation/text_personas \
    --variant full
```

#### `convert_question_json_to_text.py`
**Purpose**: Converts survey question JSON files to text format

**What it does**:
- Reads question blocks from documentation/JSON
- Converts questions to text format for LLM prompts
- Saves to `text_simulation/text_questions/`

#### `create_text_simulation_input.py`
**Purpose**: Combines personas with questions to create simulation inputs

**What it does**:
- Pairs each persona with questions they need to answer
- Creates input files that combine persona profile + question text
- Saves to `text_simulation/text_simulation_input/`

#### `run_LLM_simulations.py`
**Purpose**: **Main simulation script** - Runs LLM to generate responses

**What it does**:
1. Loads persona-question pairs from input files
2. Sends prompts to LLM (OpenAI GPT models)
3. LLM generates responses based on persona profile
4. Saves responses to `text_simulation/text_simulation_output/`
5. Verifies and validates responses

**Key features**:
- Async batch processing for efficiency
- Retry logic for failed requests
- Response verification
- Configurable via `text_simulation/configs/openai_config.yaml`

**Usage**:
```bash
python text_simulation/run_LLM_simulations.py \
    --config text_simulation/configs/openai_config.yaml \
    --max_personas 5  # Optional: limit for testing
```

#### `llm_helper.py`
**Purpose**: Helper functions for LLM API interactions

**What it does**:
- Manages API calls to OpenAI/other LLM providers
- Handles rate limiting and retries
- Batch processing utilities
- Error handling

#### `postprocess_responses.py`
**Purpose**: Post-processes and validates LLM responses

**What it does**:
- Validates response format
- Extracts structured data from LLM outputs
- Handles edge cases and errors

---

### 📊 **Evaluation Pipeline** (`evaluation/`)

These scripts evaluate how well the digital twins perform compared to real responses.

#### `json2csv.py`
**Purpose**: Converts JSON answer blocks to CSV format for analysis

**What it does**:
- Converts simulation outputs (JSON) to CSV format
- Aligns with ground truth data structure
- Handles column mapping via `column_mapping.csv`

#### `mad_accuracy_evaluation.py`
**Purpose**: **Main evaluation script** - Computes accuracy metrics

**What it does**:
1. Compares LLM-simulated responses vs. real responses
2. Computes **MAD (Mean Absolute Difference)** metrics
3. Calculates correlations between simulated and real responses
4. Generates Excel summaries and visualization plots
5. Analyzes accuracy by question type, wave, etc.

**Metrics computed**:
- Mean Absolute Difference (MAD)
- Correlation coefficients
- Confidence intervals
- Decile analysis

**Output**: Excel files and plots in trial directory

#### `within_between_subjects.py`
**Purpose**: Statistical analysis comparing within-subject vs. between-subject consistency

**What it does**:
- Analyzes how consistent responses are:
  - **Within-subject**: Same person across different questions
  - **Between-subject**: Different people on same questions
- Tests if digital twins maintain individual consistency

#### `pricing_analysis.py`
**Purpose**: Analyzes pricing-related questions and responses

**What it does**:
- Specialized analysis for economic/pricing questions
- Compares willingness-to-pay (WTP) and willingness-to-accept (WTA)
- Evaluates pricing decision consistency

---

### 🤖 **ML Prediction** (`ml_prediction/`)

Alternative approach using traditional machine learning instead of LLMs.

#### `predict_answer_xgboost.py`
**Purpose**: Trains XGBoost models to predict survey responses

**What it does**:
- Uses XGBoost (gradient boosting) instead of LLMs
- Trains models using cross-validation
- Hyperparameter tuning
- Generates predictions for comparison with LLM approach

**Usage**:
```bash
python ml_prediction/predict_answer_xgboost.py \
    --config ml_prediction/ml_prediction_config.yaml \
    --cv-folds 3
```

#### `prepare_xgboost_for_mad.py`
**Purpose**: Formats XGBoost predictions for MAD evaluation

**What it does**:
- Converts XGBoost predictions to format compatible with evaluation pipeline
- Can run MAD evaluation directly

---

### 🚀 **Pipeline Scripts** (`scripts/`)

Convenience scripts that run multiple steps in sequence.

#### `run_pipeline.sh`
**Purpose**: Runs the complete simulation pipeline

**What it does** (in order):
1. Converts personas to text format
2. Converts questions to text format
3. Creates simulation input files
4. Runs LLM simulations

**Usage**:
```bash
# Test with 5 personas
./scripts/run_pipeline.sh --max_personas=5

# Run all 2058 personas
./scripts/run_pipeline.sh
```

#### `run_evaluation_pipeline.sh`
**Purpose**: Runs the complete evaluation pipeline

**What it does** (in order):
1. Converts JSON to CSV
2. Runs MAD accuracy evaluation
3. Runs within-between subjects analysis
4. Runs pricing analysis

**Usage**:
```bash
./scripts/run_evaluation_pipeline.sh
```

---

### 📓 **Notebooks** (`notebooks/`)

Interactive Jupyter notebooks for exploration and demos.

#### `demo_simple_simulation.ipynb`
**Purpose**: Quick start demo for simulating responses

**What it does**:
- Loads personas from Hugging Face (no local setup needed)
- Shows how to create custom questions
- Demonstrates batch simulation
- Works in Google Colab

#### `demo_full_pipeline.ipynb`
**Purpose**: Complete pipeline walkthrough in notebook format

**What it does**:
- Full workflow from data prep to evaluation
- Alternative to shell scripts for interactive exploration

---

## Typical Workflow

### 1. **Setup**
```bash
# Download dataset
python download_dataset.py

# Install dependencies (if using Poetry)
poetry install

# Set up API key in .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 2. **Run Simulation**
```bash
# Quick test
./scripts/run_pipeline.sh --max_personas=5

# Full run
./scripts/run_pipeline.sh
```

### 3. **Evaluate Results**
```bash
./scripts/run_evaluation_pipeline.sh
```

### 4. **View Results**
- Simulation outputs: `text_simulation/text_simulation_output/`
- Evaluation results: Trial directory specified in `evaluation/evaluation_basic.yaml`

---

## Key Concepts

### **Persona**
A text representation of a person based on their survey responses. Contains:
- Demographics
- Personality traits
- Past responses to questions
- Cognitive test results
- Economic preferences

### **Wave**
The dataset is organized into 4 waves of surveys:
- **Wave 1**: Demographics, personality, cognitive tests, economic preferences
- **Wave 2**: Additional cognitive tests, decision-making scenarios
- **Wave 3**: More economic preferences, probability matching
- **Wave 4**: New questions (used for testing - these weren't in original surveys)

### **Digital Twin**
The LLM-generated simulation of a person. Given a persona profile, the LLM predicts how that person would respond to new questions.

### **MAD (Mean Absolute Difference)**
Primary evaluation metric. Measures how different simulated responses are from real responses. Lower MAD = better accuracy.

---

## Configuration Files

- `text_simulation/configs/openai_config.yaml` - LLM simulation settings
- `evaluation/evaluation_basic.yaml` - Evaluation pipeline settings
- `ml_prediction/ml_prediction_config.yaml` - XGBoost model settings

---

## Output Locations

- **Simulation outputs**: `text_simulation/text_simulation_output/`
- **Evaluation results**: Trial directory (specified in evaluation config)
- **ML predictions**: `ml_prediction/output/`
- **Raw data**: `data/`

---

## Use Cases

1. **Research**: Test new survey questions on digital twins before fielding to real people
2. **Benchmarking**: Compare different LLM models for persona simulation
3. **Analysis**: Understand how well LLMs can capture human decision-making patterns
4. **Prediction**: Predict how groups of people might respond to new questions

---

## Dependencies

Main packages:
- `datasets` - Hugging Face dataset loading
- `openai` - OpenAI API access
- `pandas` - Data manipulation
- `xgboost` - ML predictions
- `scikit-learn` - ML utilities
- `matplotlib` - Plotting
- `tqdm` - Progress bars

See `pyproject.toml` for complete list.


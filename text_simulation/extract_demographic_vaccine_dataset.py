"""
Extract Demographics and Vaccine Answer Dataset

This script extracts demographic information (first 14 questions from Wave 1)
and the vaccine question answer (QID291) from persona JSON files, creating
a clean dataset for vaccine prediction tasks.

Usage:
    python extract_demographic_vaccine_dataset.py --input_dir ./data/mega_persona_json/mega_persona --output_file ./data/demographic_vaccine_dataset.json
"""

import os
import json
import argparse
import re
from tqdm import tqdm
from typing import Dict, List, Optional, Any
from pathlib import Path


def strip_html(text: Any) -> str:
    """Strip HTML tags from text and normalize whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    
    text = re.sub(r'<[^>]*>', ' ', text)
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_persona_id(filename: str) -> Optional[str]:
    """Extract persona ID from filename."""
    match = re.search(r'pid_(\d+)', filename)
    return match.group(1) if match else None


def _recursively_extract_questions(elements_list: List[Dict], questions: List[Dict]) -> None:
    """Recursively extract all questions from JSON structure."""
    for element in elements_list:
        element_type = element.get("ElementType")
        if element_type == "Block":
            if element.get("Questions"):
                questions.extend(element["Questions"])
        elif element_type == "Branch":
            if element.get("Elements"):
                _recursively_extract_questions(element["Elements"], questions)


def extract_demographics(questions: List[Dict]) -> Dict[str, Any]:
    """
    Extract demographic questions (QID11-QID24) from questions list.
    
    Demographic questions are typically the first 14 questions in Wave 1.
    """
    demographics = {}
    
    # Demographic QIDs in order: QID11-QID24 (14 questions)
    demographic_qids = [
        "QID11",  # Region
        "QID12",  # Sex
        "QID13",  # Age
        "QID14",  # Education
        "QID15",  # Race/Origin
        "QID16",  # Citizenship
        "QID17",  # Marital Status
        "QID18",  # Religion
        "QID19",  # Religious Attendance
        "QID20",  # Political Affiliation
        "QID21",  # Income
        "QID22",  # Political Views
        "QID23",  # Household Size
        "QID24",  # Employment
    ]
    
    # Create a mapping of QID to question data
    qid_to_question = {}
    for question in questions:
        qid = question.get("QuestionID")
        if qid:
            qid_to_question[qid] = question
    
    # Extract demographic answers
    for qid in demographic_qids:
        if qid in qid_to_question:
            question = qid_to_question[qid]
            question_text = strip_html(question.get("QuestionText", ""))
            answers = question.get("Answers", {})
            
            # Extract answer based on question type
            answer_value = None
            answer_text = None
            
            if question.get("QuestionType") == "MC":
                selected_positions = answers.get("SelectedByPosition")
                selected_texts = answers.get("SelectedText")
                
                if selected_positions is not None:
                    # Handle both single choice and multiple choice
                    if isinstance(selected_positions, list):
                        if len(selected_positions) > 0:
                            answer_value = selected_positions[0]
                            answer_text = selected_texts[0] if selected_texts and len(selected_texts) > 0 else None
                    else:
                        answer_value = selected_positions
                        answer_text = selected_texts if isinstance(selected_texts, str) else (selected_texts[0] if selected_texts else None)
            
            demographics[qid] = {
                "question_text": question_text,
                "answer_value": answer_value,
                "answer_text": strip_html(answer_text) if answer_text else None
            }
    
    return demographics


def extract_vaccine_answer(questions: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Extract QID291 (vaccine question) answer.
    
    QID291 is the omission bias vaccine question from Wave 1.
    """
    for question in questions:
        qid = question.get("QuestionID")
        if qid == "QID291":
            question_text = strip_html(question.get("QuestionText", ""))
            answers = question.get("Answers", {})
            
            # Extract answer (Single Choice, 1-4 scale)
            selected_position = answers.get("SelectedByPosition")
            selected_text = answers.get("SelectedText")
            
            # Handle both single value and list
            if isinstance(selected_position, list) and len(selected_position) > 0:
                selected_position = selected_position[0]
                selected_text = selected_text[0] if isinstance(selected_text, list) and len(selected_text) > 0 else selected_text
            elif selected_position is None:
                return None
            
            return {
                "question_id": "QID291",
                "question_text": question_text,
                "answer_value": int(selected_position) if selected_position else None,
                "answer_text": strip_html(selected_text) if selected_text else None
            }
    
    return None


def format_demographic_description(demographics: Dict[str, Any]) -> str:
    """
    Format demographics as a concise natural language description.
    
    Example: "This person is 30-49 years old, lives in the South region..."
    """
    parts = []
    
    # Age
    if "QID13" in demographics and demographics["QID13"]["answer_text"]:
        parts.append(f"{demographics['QID13']['answer_text']} years old")
    
    # Region
    if "QID11" in demographics and demographics["QID11"]["answer_text"]:
        parts.append(f"lives in the {demographics['QID11']['answer_text']} region")
    
    # Sex/Gender
    if "QID12" in demographics and demographics["QID12"]["answer_text"]:
        parts.append(f"identifies as {demographics['QID12']['answer_text']}")
    
    # Education
    if "QID14" in demographics and demographics["QID14"]["answer_text"]:
        parts.append(f"has {demographics['QID14']['answer_text']} education")
    
    # Race/Origin
    if "QID15" in demographics and demographics["QID15"]["answer_text"]:
        parts.append(f"is {demographics['QID15']['answer_text']}")
    
    # Citizenship
    if "QID16" in demographics and demographics["QID16"]["answer_text"]:
        parts.append(f"is {demographics['QID16']['answer_text']} a U.S. citizen")
    
    # Marital Status
    if "QID17" in demographics and demographics["QID17"]["answer_text"]:
        parts.append(f"is {demographics['QID17']['answer_text']}")
    
    # Religion
    if "QID18" in demographics and demographics["QID18"]["answer_text"]:
        parts.append(f"follows {demographics['QID18']['answer_text']}")
    
    # Religious Attendance
    if "QID19" in demographics and demographics["QID19"]["answer_text"]:
        parts.append(f"attends religious services {demographics['QID19']['answer_text']}")
    
    # Political Affiliation
    if "QID20" in demographics and demographics["QID20"]["answer_text"]:
        parts.append(f"identifies as {demographics['QID20']['answer_text']}")
    
    # Income
    if "QID21" in demographics and demographics["QID21"]["answer_text"]:
        parts.append(f"has family income {demographics['QID21']['answer_text']}")
    
    # Political Views
    if "QID22" in demographics and demographics["QID22"]["answer_text"]:
        parts.append(f"describes political views as {demographics['QID22']['answer_text']}")
    
    # Household Size
    if "QID23" in demographics and demographics["QID23"]["answer_text"]:
        parts.append(f"lives in a household of {demographics['QID23']['answer_text']}")
    
    # Employment
    if "QID24" in demographics and demographics["QID24"]["answer_text"]:
        parts.append(f"is currently {demographics['QID24']['answer_text']}")
    
    if parts:
        return "This person is " + ", ".join(parts) + "."
    else:
        return "This person's demographic information is not available."


def process_persona_file(json_path: str) -> Optional[Dict[str, Any]]:
    """Process a single persona JSON file and extract demographics + vaccine answer."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Handle JSON that might be wrapped in quotes
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1].replace('\\"', '"').replace('\\\\', '\\')
            data = json.loads(content)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None
    
    # Extract persona ID
    persona_id = extract_persona_id(os.path.basename(json_path))
    if not persona_id:
        print(f"Warning: Could not extract persona ID from {json_path}")
        return None
    
    # Extract all questions
    questions = []
    if isinstance(data, list):
        _recursively_extract_questions(data, questions)
    elif isinstance(data, dict):
        if "Elements" in data:
            _recursively_extract_questions(data["Elements"], questions)
        elif "Questions" in data:
            questions = data["Questions"]
    
    # Extract demographics
    demographics = extract_demographics(questions)
    
    # Extract vaccine answer
    vaccine_answer = extract_vaccine_answer(questions)
    
    # Only return if we have both demographics and vaccine answer
    if not demographics or vaccine_answer is None:
        return None
    
    # Format demographic description
    demographic_description = format_demographic_description(demographics)
    
    return {
        "persona_id": f"pid_{persona_id}",
        "demographics": demographics,
        "demographic_description": demographic_description,
        "vaccine_question": {
            "question_id": vaccine_answer["question_id"],
            "question_text": vaccine_answer["question_text"],
            "actual_answer": vaccine_answer["answer_value"],
            "actual_answer_text": vaccine_answer["answer_text"]
        }
    }


def extract_dataset(input_dir: str, output_file: str) -> None:
    """
    Extract demographics and vaccine answers from all persona JSON files.
    
    Args:
        input_dir: Directory containing persona JSON files
        output_file: Path to save the extracted dataset (JSON)
    """
    # Find all persona JSON files
    json_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('_mega_persona.json'):
            json_files.append(os.path.join(input_dir, filename))
    
    if not json_files:
        print(f"No persona JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} persona files. Processing...")
    
    # Process each file
    dataset = []
    failed_count = 0
    
    for json_path in tqdm(json_files, desc="Extracting data"):
        result = process_persona_file(json_path)
        if result:
            dataset.append(result)
        else:
            failed_count += 1
    
    print(f"Successfully extracted data from {len(dataset)} personas.")
    if failed_count > 0:
        print(f"Failed to extract from {failed_count} personas (missing demographics or vaccine answer).")
    
    # Save dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_file}")
    print(f"Total entries: {len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract demographics and vaccine answers from persona JSON files."
    )
    parser.add_argument(
        "--input_dir",
        default="./data/mega_persona_json/mega_persona",
        help="Directory containing persona JSON files"
    )
    parser.add_argument(
        "--output_file",
        default="./data/demographic_vaccine_dataset.json",
        help="Output JSON file path for the extracted dataset"
    )
    
    args = parser.parse_args()
    
    extract_dataset(args.input_dir, args.output_file)


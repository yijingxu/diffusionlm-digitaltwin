"""
Extract Demographics, Psychographics, and Prediction Variables Dataset

This script extracts:
- Demographics (QID11-QID24): All demographic questions
- Predictor Variables:
  - QID26: Need for Cognition (Psychographic)
  - QID31: Spendthrift/Tightwad (Consumer Trait)
  - QID239: Maximization (Psychographics)
  - QID35: Minimalism (Lifestyle)
- Predicted Variables:
  - QID34: The Impulse Test
  - QID36: Financial Literacy
  - QID117: The Trust Game (sender)
  - QID250: Risk Aversion
  - QID150: Mental Accounting

Usage:
    python extract_demographic_vaccine_dataset.py --input_dir ./data/mega_persona_json/mega_persona --output_file ./data/demographic_games_dataset.csv
"""

import os
import json
import csv
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


def extract_single_choice_answer(question: Dict) -> Optional[Dict[str, Any]]:
    """
    Extract answer from a single choice (MC) question.
    
    Returns dict with question_id, question_text, answer_value, answer_text
    """
    qid = question.get("QuestionID")
    question_text = strip_html(question.get("QuestionText", ""))
    answers = question.get("Answers", {})
    
    selected_position = answers.get("SelectedByPosition")
    selected_text = answers.get("SelectedText")
    
    # Handle both single value and list
    if isinstance(selected_position, list) and len(selected_position) > 0:
        selected_position = selected_position[0]
        selected_text = selected_text[0] if isinstance(selected_text, list) and len(selected_text) > 0 else selected_text
    elif selected_position is None:
        return None
    
    return {
        "question_id": qid,
        "question_text": question_text,
        "answer_value": int(selected_position) if selected_position else None,
        "answer_text": strip_html(selected_text) if selected_text else None
    }


def extract_vaccine_answer(questions: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Extract QID291 (vaccine question) answer.
    
    QID291 is the omission bias vaccine question (appears in Wave 1 and/or Wave 4).
    """
    for question in questions:
        qid = question.get("QuestionID")
        if qid == "QID291":
            answer_data = extract_single_choice_answer(question)
            # Check if answer exists (question might be present but unanswered)
            if answer_data and answer_data.get("answer_value") is not None:
                return answer_data
            # Question exists but no answer
            return None
    return None


def extract_question_by_qid(questions: List[Dict], target_qid: str, debug: bool = False) -> Optional[Dict[str, Any]]:
    """
    Extract answer for a specific QID.
    
    Args:
        questions: List of question dictionaries
        target_qid: The QuestionID to extract (e.g., "QID174", "QID290", "QID157")
        debug: If True, print debug info when question found but no answer
    
    Returns:
        Dict with question_id, question_text, answer_value, answer_text, or None if not found
    """
    for question in questions:
        qid = question.get("QuestionID")
        if qid == target_qid:
            answer_data = extract_single_choice_answer(question)
            # Check if answer exists (question might be present but unanswered)
            if answer_data and answer_data.get("answer_value") is not None:
                return answer_data
            # Question exists but no answer
            if debug:
                answers = question.get("Answers", {})
                print(f"  Debug: Found {target_qid} but no answer. Answers dict: {answers}")
            return None
    return None


def extract_ultimatum_game(questions: List[Dict]) -> Dict[str, Any]:
    """
    Extract Ultimatum game questions (QID224 sender, QID225-QID230 receiver).
    
    Returns dict with sender answer and receiver answers for different offers.
    """
    result = {
        "sender": None,  # QID224
        "receiver": {}   # QID225-QID230
    }
    
    for question in questions:
        qid = question.get("QuestionID")
        if qid == "QID224":
            result["sender"] = extract_single_choice_answer(question)
        elif qid in ["QID225", "QID226", "QID227", "QID228", "QID229", "QID230"]:
            answer_data = extract_single_choice_answer(question)
            if answer_data:
                result["receiver"][qid] = answer_data
    
    return result


def extract_trust_game(questions: List[Dict]) -> Dict[str, Any]:
    """
    Extract Trust game questions (QID117 sender, QID118-QID122 receiver).
    
    Returns dict with sender answer and receiver answers for different amounts.
    """
    result = {
        "sender": None,  # QID117
        "receiver": {}   # QID118-QID122
    }
    
    for question in questions:
        qid = question.get("QuestionID")
        if qid == "QID117":
            result["sender"] = extract_single_choice_answer(question)
        elif qid in ["QID118", "QID119", "QID120", "QID121", "QID122"]:
            answer_data = extract_single_choice_answer(question)
            if answer_data:
                result["receiver"][qid] = answer_data
    
    return result


def extract_dictator_game(questions: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Extract Dictator game question (QID231).
    
    QID231 is the dictator game from Wave 3.
    """
    for question in questions:
        qid = question.get("QuestionID")
        if qid == "QID231":
            return extract_single_choice_answer(question)
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
    
    # Extract Trust Game (QID117 is a predicted variable)
    trust_game = extract_trust_game(questions)
    
    # Extract predictor variables
    qid26_answer = extract_question_by_qid(questions, "QID26")  # Need for Cognition
    qid31_answer = extract_question_by_qid(questions, "QID31")  # Spendthrift/Tightwad
    qid239_answer = extract_question_by_qid(questions, "QID239")  # Maximization
    qid35_answer = extract_question_by_qid(questions, "QID35")  # Minimalism
    
    # Extract predicted variables
    qid34_answer = extract_question_by_qid(questions, "QID34")  # The Impulse Test
    qid36_answer = extract_question_by_qid(questions, "QID36")  # Financial Literacy
    qid250_answer = extract_question_by_qid(questions, "QID250")  # Risk Aversion
    qid150_answer = extract_question_by_qid(questions, "QID150")  # Mental Accounting
    # QID117 (Trust Game sender) is already extracted above
    
    # Only return if we have demographics and at least one game answer
    if not demographics:
        return None
    
    # Format demographic description
    demographic_description = format_demographic_description(demographics)
    
    result = {
        "persona_id": f"pid_{persona_id}",
        "demographics": demographics,
        "demographic_description": demographic_description,
    }
    
    # Add predictor variables if available
    if qid26_answer:
        result["qid26_question"] = {
            "question_id": qid26_answer["question_id"],
            "question_text": qid26_answer["question_text"],
            "actual_answer": qid26_answer["answer_value"],
            "actual_answer_text": qid26_answer["answer_text"]
        }
    
    if qid31_answer:
        result["qid31_question"] = {
            "question_id": qid31_answer["question_id"],
            "question_text": qid31_answer["question_text"],
            "actual_answer": qid31_answer["answer_value"],
            "actual_answer_text": qid31_answer["answer_text"]
        }
    
    if qid239_answer:
        result["qid239_question"] = {
            "question_id": qid239_answer["question_id"],
            "question_text": qid239_answer["question_text"],
            "actual_answer": qid239_answer["answer_value"],
            "actual_answer_text": qid239_answer["answer_text"]
        }
    
    if qid35_answer:
        result["qid35_question"] = {
            "question_id": qid35_answer["question_id"],
            "question_text": qid35_answer["question_text"],
            "actual_answer": qid35_answer["answer_value"],
            "actual_answer_text": qid35_answer["answer_text"]
        }
    
    # Add predicted variables if available
    if qid34_answer:
        result["qid34_question"] = {
            "question_id": qid34_answer["question_id"],
            "question_text": qid34_answer["question_text"],
            "actual_answer": qid34_answer["answer_value"],
            "actual_answer_text": qid34_answer["answer_text"]
        }
    
    if qid36_answer:
        result["qid36_question"] = {
            "question_id": qid36_answer["question_id"],
            "question_text": qid36_answer["question_text"],
            "actual_answer": qid36_answer["answer_value"],
            "actual_answer_text": qid36_answer["answer_text"]
        }
    
    if qid250_answer:
        result["qid250_question"] = {
            "question_id": qid250_answer["question_id"],
            "question_text": qid250_answer["question_text"],
            "actual_answer": qid250_answer["answer_value"],
            "actual_answer_text": qid250_answer["answer_text"]
        }
    
    if qid150_answer:
        result["qid150_question"] = {
            "question_id": qid150_answer["question_id"],
            "question_text": qid150_answer["question_text"],
            "actual_answer": qid150_answer["answer_value"],
            "actual_answer_text": qid150_answer["answer_text"]
        }
    
    # Add trust game if available (QID117 is a predicted variable)
    if trust_game.get("sender") or trust_game.get("receiver"):
        result["trust_game"] = {
            "sender": {
                "question_id": trust_game["sender"]["question_id"] if trust_game["sender"] else None,
                "question_text": trust_game["sender"]["question_text"] if trust_game["sender"] else None,
                "actual_answer": trust_game["sender"]["answer_value"] if trust_game["sender"] else None,
                "actual_answer_text": trust_game["sender"]["answer_text"] if trust_game["sender"] else None
            } if trust_game["sender"] else None,
            "receiver": {
                qid: {
                    "question_id": data["question_id"],
                    "question_text": data["question_text"],
                    "actual_answer": data["answer_value"],
                    "actual_answer_text": data["answer_text"]
                }
                for qid, data in trust_game["receiver"].items()
            } if trust_game["receiver"] else {}
        }
    
    # Only return if we have at least one predictor or predicted variable
    has_predictors = any([qid26_answer, qid31_answer, qid239_answer, qid35_answer])
    has_predicted = any([qid34_answer, qid36_answer, trust_game.get("sender"), qid250_answer, qid150_answer])
    
    if not (has_predictors or has_predicted):
        return None
    
    return result


def extract_question_metadata(question: Dict) -> Dict[str, Any]:
    """
    Extract metadata about a question (QID, text, type, options).
    
    Returns a dictionary with question information.
    """
    qid = question.get("QuestionID", "")
    question_text = strip_html(question.get("QuestionText", ""))
    question_type = question.get("QuestionType", "")
    
    # Extract options/choices based on question type
    options = []
    options_text = ""
    
    if question_type == "MC":
        # Multiple Choice - can have Options or Columns
        options_list = question.get("Options", []) or question.get("Columns", [])
        if options_list:
            for i, opt in enumerate(options_list, 1):
                opt_text = strip_html(opt) if isinstance(opt, str) else strip_html(str(opt))
                options.append(f"{i} - {opt_text}")
            options_text = "\n".join(options)
        else:
            options_text = "Multiple Choice (no options found)"
    elif question_type == "Matrix":
        # Matrix - has Rows and Columns
        rows = question.get("Rows", [])
        columns = question.get("Columns", [])
        if columns:
            col_list = []
            for i, col in enumerate(columns, 1):
                col_text = strip_html(col) if isinstance(col, str) else strip_html(str(col))
                col_list.append(f"{i} = {col_text}")
            options_text = "Columns:\n" + "\n".join(col_list)
            if rows:
                row_list = []
                for i, row in enumerate(rows[:10], 1):  # Show first 10 rows
                    row_text = strip_html(row) if isinstance(row, str) else strip_html(str(row))
                    row_list.append(f"{i}. {row_text}")
                if len(rows) > 10:
                    row_list.append(f"... and {len(rows) - 10} more rows")
                options_text += "\n\nRows:\n" + "\n".join(row_list)
        else:
            options_text = "Matrix (no columns found)"
    elif question_type == "Slider":
        # Slider - usually has min/max
        min_val = question.get("MinValue", "")
        max_val = question.get("MaxValue", "")
        if min_val or max_val:
            options_text = f"Range: {min_val} to {max_val}"
        else:
            options_text = "Slider (no range specified)"
    elif question_type == "TE":
        # Text Entry
        options_text = "Text Entry"
    elif question_type == "DB":
        # Descriptive/Block
        options_text = "Descriptive (no answer required)"
    else:
        # Try to find any options/columns
        options_list = question.get("Options", []) or question.get("Columns", [])
        if options_list:
            for i, opt in enumerate(options_list, 1):
                opt_text = strip_html(opt) if isinstance(opt, str) else strip_html(str(opt))
                options.append(f"{i} - {opt_text}")
            options_text = "\n".join(options)
        else:
            options_text = f"Type: {question_type} (no options found)"
    
    return {
        "qid": qid,
        "question_text": question_text,
        "question_type": question_type,
        "options": options_text
    }


def extract_all_questions_catalog(input_dir: str, output_file: str) -> None:
    """
    Extract a catalog of all unique questions with their QIDs, text, and choices.
    
    Args:
        input_dir: Directory containing persona JSON files
        output_file: Path to save the question catalog (CSV format)
    """
    json_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('_mega_persona.json'):
            json_files.append(os.path.join(input_dir, filename))
    
    if not json_files:
        print(f"No persona JSON files found in {input_dir}")
        return
    
    print(f"Extracting question catalog from {len(json_files)} persona files...")
    
    # Dictionary to store unique questions (QID -> question metadata)
    questions_catalog = {}
    
    # Process files to collect all unique questions
    for json_path in tqdm(json_files, desc="Cataloging questions"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1].replace('\\"', '"').replace('\\\\', '\\')
                data = json.loads(content)
        except Exception as e:
            continue
        
        # Extract all questions
        questions = []
        if isinstance(data, list):
            _recursively_extract_questions(data, questions)
        elif isinstance(data, dict):
            if "Elements" in data:
                _recursively_extract_questions(data["Elements"], questions)
            elif "Questions" in data:
                questions = data["Questions"]
        
        # Add questions to catalog
        for question in questions:
            qid = question.get("QuestionID")
            if qid and qid not in questions_catalog:
                questions_catalog[qid] = extract_question_metadata(question)
    
    # Sort by QID
    sorted_qids = sorted(questions_catalog.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['qid', 'question_type', 'question_text', 'options']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for qid in sorted_qids:
            q_data = questions_catalog[qid]
            writer.writerow({
                'qid': q_data['qid'],
                'question_type': q_data['question_type'],
                'question_text': q_data['question_text'],
                'options': q_data['options']
            })
    
    print(f"Question catalog saved to {output_file}")
    print(f"Total unique questions found: {len(questions_catalog)}")


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
    
    # Counters for predictor variables
    qid26_found_count = 0
    qid31_found_count = 0
    qid239_found_count = 0
    qid35_found_count = 0
    
    # Counters for predicted variables
    qid34_found_count = 0
    qid36_found_count = 0
    qid117_found_count = 0
    qid250_found_count = 0
    qid150_found_count = 0
    
    # Track which QIDs are present (even without answers) for first few files
    sample_qids_found = set()
    all_sample_qids = set()  # All QIDs found in sample
    all_qids_ever_found = set()  # All QIDs found across ALL files
    target_qids_found_anywhere = {
        "QID26": False, "QID31": False, "QID239": False, "QID35": False,  # Predictors
        "QID34": False, "QID36": False, "QID117": False, "QID250": False, "QID150": False  # Predicted
    }
    sample_checked = 0
    max_sample_check = 10
    
    for json_path in tqdm(json_files, desc="Extracting data"):
        # Check for target QIDs in this file
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1].replace('\\"', '"').replace('\\\\', '\\')
                data = json.loads(content)
            
            questions = []
            if isinstance(data, list):
                _recursively_extract_questions(data, questions)
            elif isinstance(data, dict):
                if "Elements" in data:
                    _recursively_extract_questions(data["Elements"], questions)
                elif "Questions" in data:
                    questions = data["Questions"]
            
            for q in questions:
                qid = q.get("QuestionID")
                if qid:
                    all_qids_ever_found.add(qid)
                    if qid in target_qids_found_anywhere:
                        target_qids_found_anywhere[qid] = True
        except:
            pass
        
        result = process_persona_file(json_path)
        if result:
            # Count predictor variables
            if result.get("qid26_question"):
                qid26_found_count += 1
            if result.get("qid31_question"):
                qid31_found_count += 1
            if result.get("qid239_question"):
                qid239_found_count += 1
            if result.get("qid35_question"):
                qid35_found_count += 1
            
            # Count predicted variables
            if result.get("qid34_question"):
                qid34_found_count += 1
            if result.get("qid36_question"):
                qid36_found_count += 1
            if result.get("trust_game", {}).get("sender"):
                qid117_found_count += 1
            if result.get("qid250_question"):
                qid250_found_count += 1
            if result.get("qid150_question"):
                qid150_found_count += 1
            
            dataset.append(result)
            
            # Sample check: see what QIDs are in the first few files
            if sample_checked < max_sample_check:
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content.startswith('"') and content.endswith('"'):
                            content = content[1:-1].replace('\\"', '"').replace('\\\\', '\\')
                        data = json.loads(content)
                    
                    questions = []
                    if isinstance(data, list):
                        _recursively_extract_questions(data, questions)
                    elif isinstance(data, dict):
                        if "Elements" in data:
                            _recursively_extract_questions(data["Elements"], questions)
                        elif "Questions" in data:
                            questions = data["Questions"]
                    
                    for q in questions:
                        qid = q.get("QuestionID")
                        if qid:
                            all_sample_qids.add(qid)
                            if qid in ["QID26", "QID31", "QID239", "QID35", 
                                      "QID34", "QID36", "QID117", "QID250", "QID150"]:
                                sample_qids_found.add(qid)
                    sample_checked += 1
                except:
                    pass
        else:
            failed_count += 1
    
    print(f"Successfully extracted data from {len(dataset)} personas.")
    print(f"\nPredictor Variables:")
    print(f"  - Found QID26 (Need for Cognition) for {qid26_found_count} personas")
    print(f"  - Found QID31 (Spendthrift/Tightwad) for {qid31_found_count} personas")
    print(f"  - Found QID239 (Maximization) for {qid239_found_count} personas")
    print(f"  - Found QID35 (Minimalism) for {qid35_found_count} personas")
    print(f"\nPredicted Variables:")
    print(f"  - Found QID34 (Impulse Test) for {qid34_found_count} personas")
    print(f"  - Found QID36 (Financial Literacy) for {qid36_found_count} personas")
    print(f"  - Found QID117 (Trust Game) for {qid117_found_count} personas")
    print(f"  - Found QID250 (Risk Aversion) for {qid250_found_count} personas")
    print(f"  - Found QID150 (Mental Accounting) for {qid150_found_count} personas")
    
    # Check if target QIDs were found anywhere
    found_qids = [qid for qid, found in target_qids_found_anywhere.items() if found]
    missing_qids = [qid for qid, found in target_qids_found_anywhere.items() if not found]
    
    if found_qids:
        print(f"\n  Found target QIDs in dataset: {found_qids}")
        print(f"  Missing target QIDs: {missing_qids}")
    else:
        print(f"\n  None of the target QIDs were found in ANY of the {len(json_files)} files.")
        print(f"  This suggests these questions are not in the persona JSON files.")
    
    if sample_qids_found:
        print(f"\n  Sample check (first {sample_checked} files): Found these target QIDs present (may not have answers): {sorted(sample_qids_found)}")
    else:
        print(f"\n  Sample check (first {sample_checked} files): None of target QIDs found in sample files.")
    
    # Show sample of available QIDs to help diagnose
    if all_sample_qids:
        sample_list = sorted(list(all_sample_qids))[:50]  # First 50 QIDs
        print(f"\n  Sample QIDs found in first file (showing first 50): {sample_list}")
        if len(all_sample_qids) > 50:
            print(f"  ... and {len(all_sample_qids) - 50} more QIDs")
    
    # Show total unique QIDs found across all files
    if all_qids_ever_found:
        sorted_all_qids = sorted(list(all_qids_ever_found))
        print(f"\n  Total unique QIDs found across all {len(json_files)} files: {len(all_qids_ever_found)}")
        print(f"  QID range: {sorted_all_qids[0]} to {sorted_all_qids[-1]}")
        
        # Check if any QIDs in the 20-300 range exist (where our targets might be)
        mid_range_qids = [qid for qid in sorted_all_qids if any(qid.startswith(f"QID{i}") for i in range(20, 301))]
        if mid_range_qids:
            print(f"  Sample QIDs in 20-300 range: {mid_range_qids[:30]}")
            if len(mid_range_qids) > 30:
                print(f"    ... and {len(mid_range_qids) - 30} more in this range")
    
    if failed_count > 0:
        print(f"Failed to extract from {failed_count} personas (missing demographics or game answers).")
    
    # Save dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine output format based on file extension
    if output_file.endswith('.csv'):
        # Save as CSV
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            if not dataset:
                print("Warning: No data to save!")
                return
            
            fieldnames = [
                'persona_id', 'demographic_description',
                # Predictor variables
                'qid26_answer', 'qid31_answer', 'qid239_answer', 'qid35_answer',
                # Predicted variables
                'qid34_answer', 'qid36_answer', 'qid117_answer', 'qid250_answer', 'qid150_answer'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in dataset:
                row = {
                    'persona_id': entry['persona_id'],
                    'demographic_description': entry.get('demographic_description', ''),
                    # Predictor variables
                    'qid26_answer': entry.get('qid26_question', {}).get('actual_answer', ''),
                    'qid31_answer': entry.get('qid31_question', {}).get('actual_answer', ''),
                    'qid239_answer': entry.get('qid239_question', {}).get('actual_answer', ''),
                    'qid35_answer': entry.get('qid35_question', {}).get('actual_answer', ''),
                    # Predicted variables
                    'qid34_answer': entry.get('qid34_question', {}).get('actual_answer', ''),
                    'qid36_answer': entry.get('qid36_question', {}).get('actual_answer', ''),
                    'qid117_answer': entry.get('trust_game', {}).get('sender', {}).get('actual_answer', ''),
                    'qid250_answer': entry.get('qid250_question', {}).get('actual_answer', ''),
                    'qid150_answer': entry.get('qid150_question', {}).get('actual_answer', '')
                }
                writer.writerow(row)
        
        print(f"Dataset saved to {output_file} (CSV format)")
    else:
        # Save as JSON (backward compatibility)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {output_file} (JSON format)")
    
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
        default="./data/demographic_games_dataset.csv",
        help="Output file path (CSV or JSON format, based on extension)"
    )
    parser.add_argument(
        "--catalog_file",
        default=None,
        help="Optional: Path to save question catalog (all QIDs with questions and choices)"
    )
    
    args = parser.parse_args()
    
    # Extract question catalog if requested
    if args.catalog_file:
        extract_all_questions_catalog(args.input_dir, args.catalog_file)
    
    extract_dataset(args.input_dir, args.output_file)


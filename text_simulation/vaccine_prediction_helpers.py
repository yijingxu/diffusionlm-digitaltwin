"""
Helper functions for economic game prediction tasks.

These functions support the simplified demographic-only prediction workflow
for vaccine, trust game, dictator game, and ultimatum game questions.
"""

import json
import re
from typing import Dict, Optional, Any


def get_vaccine_question_text() -> str:
    """
    Get the vaccine question (QID291) text and format instructions.
    """
    return """Imagine that there will be a deadly flu going around your area next winter. Your doctor says that you have a 10% chance (10 out of 100) of dying from this flu. However, a new flu vaccine has been developed and tested. If taken, the vaccine prevents you from catching the deadly flu. However, there is one serious risk involved with taking this vaccine. The vaccine is made from a somewhat weaker type of flu virus, and there is a 5% (5 out of 100) risk of the vaccine causing you to die from the weaker type of flu. Imagine that this vaccine is completely covered by health insurance. If you had to decide now, which would you choose?

Options:
1 - I would definitely not take the vaccine. I would thus accept the 10% chance of dying from this flu.
2 - I would probably not take the vaccine. I would thus accept the 10% chance of dying from this flu.
3 - I would probably take the vaccine. I would thus accept the 5% chance of dying from the weaker flu in the vaccine
4 - I would definitely take the vaccine. I would thus accept the 5% chance of dying from the weaker flu in the vaccine.

## Instructions:
Based on the demographic profile above, predict on a scale of 1-4 how likely this person is to accept the vaccine.
Respond with only a single number (1, 2, 3, or 4)."""


def get_ultimatum_sender_question_text() -> str:
    """Get Ultimatum game sender question (QID224) text."""
    return """Suppose you were given $5 and had to offer to another (anonymous) person a way to split the money. You would propose how much of this money to keep for yourself and how much to send them. Then, the other person would have to decide whether or not to accept your offer. If they accept your offer, you would each receive the amount specified in your offer. If they reject your offer, you would both receive nothing. In this scenario, how much would propose to keep for yourself and how much would you propose to send to the other person?

Options:
1 - $0 for myself, $5 to the other person.
2 - $1 for myself, $4 to the other person.
3 - $2 for myself, $3 to the other person.
4 - $3 for myself, $2 to the other person.
5 - $4 for myself, $1 to the other person.
6 - $5 for myself, $0 to the other person.

## Instructions:
Based on the demographic profile above, predict which option (1-6) this person would choose as the sender.
Respond with only a single number (1, 2, 3, 4, 5, or 6)."""


def get_trust_sender_question_text() -> str:
    """Get Trust game sender question (QID117) text."""
    return """Suppose you were given $5 and had to decide how much of this money to keep for yourself and how much to send to another (anonymous) person. Any amount you send to the other person would then be tripled. That is, if you send $1, this becomes $3. If you send $2, this becomes $6, etc. Then, the other person would have to decide how much of that money to keep and how much to return to you. That is, if you send $1, this would become $3 and the other person would have to decide how much of this $3 to keep for themself and how much to send back to you. In this scenario, how much would keep for yourself and how much would you send to the other person?

Options:
1 - I would keep $0 for myself and send $5 to the other person.
2 - I would keep $1 for myself and send $4 to the other person.
3 - I would keep $2 for myself and send $3 to the other person.
4 - I would keep $3 for myself and send $2 to the other person.
5 - I would keep $4 for myself and send $1 to the other person.
6 - I would keep $5 for myself and send $0 to the other person.

## Instructions:
Based on the demographic profile above, predict which option (1-6) this person would choose as the sender.
Respond with only a single number (1, 2, 3, 4, 5, or 6)."""


def get_dictator_question_text() -> str:
    """Get Dictator game question (QID231) text."""
    return """Suppose you were given $5 and had to split the money between yourself and another (anonymous) person. You and you only would decide how to split the money, the other person would need to accept your offer. In this scenario, how much would keep for yourself and how much would you send to the other person?

Options:
1 - $0 for myself, $5 to the other person.
2 - $1 for myself, $4 to the other person.
3 - $2 for myself, $3 to the other person.
4 - $3 for myself, $2 to the other person.
5 - $4 for myself, $1 to the other person.
6 - $5 for myself, $0 to the other person.

## Instructions:
Based on the demographic profile above, predict which option (1-6) this person would choose.
Respond with only a single number (1, 2, 3, 4, 5, or 6)."""


def format_dictator_answer(dictator_answer_value: int) -> str:
    """Format dictator game answer as text."""
    answer_map = {
        1: "$0 for myself, $5 to the other person",
        2: "$1 for myself, $4 to the other person",
        3: "$2 for myself, $3 to the other person",
        4: "$3 for myself, $2 to the other person",
        5: "$4 for myself, $1 to the other person",
        6: "$5 for myself, $0 to the other person"
    }
    return answer_map.get(dictator_answer_value, f"Option {dictator_answer_value}")


def format_ultimatum_answer(ultimatum_answer_value: int) -> str:
    """Format ultimatum game answer as text."""
    answer_map = {
        1: "$0 for myself, $5 to the other person",
        2: "$1 for myself, $4 to the other person",
        3: "$2 for myself, $3 to the other person",
        4: "$3 for myself, $2 to the other person",
        5: "$4 for myself, $1 to the other person",
        6: "$5 for myself, $0 to the other person"
    }
    return answer_map.get(ultimatum_answer_value, f"Option {ultimatum_answer_value}")


def create_trust_sender_prompt_with_context(
    demographic_description: str,
    dictator_answer: Optional[int] = None,
    ultimatum_answer: Optional[int] = None
) -> str:
    """
    Create a prompt for Trust game sender prediction using demographics + dictator + ultimatum answers.
    
    Args:
        demographic_description: Natural language description of demographics
        dictator_answer: Dictator game answer (1-6) or None
        ultimatum_answer: Ultimatum game sender answer (1-6) or None
        
    Returns:
        Complete prompt string
    """
    context_parts = [f"## Demographic Profile:\n{demographic_description}"]
    
    # Add dictator game answer if provided
    if dictator_answer is not None:
        dictator_text = get_dictator_question_text().split("## Instructions:")[0].strip()
        dictator_answer_text = format_dictator_answer(dictator_answer)
        context_parts.append(f"## Dictator Game:\n{dictator_text}\n\nThis person chose: Option {dictator_answer} - {dictator_answer_text}")
    
    # Add ultimatum game answer if provided
    if ultimatum_answer is not None:
        ultimatum_text = get_ultimatum_sender_question_text().split("## Instructions:")[0].strip()
        ultimatum_answer_text = format_ultimatum_answer(ultimatum_answer)
        context_parts.append(f"## Ultimatum Game (Sender):\n{ultimatum_text}\n\nThis person chose: Option {ultimatum_answer} - {ultimatum_answer_text}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""{context}

---

## Question to Predict:
{get_trust_sender_question_text().replace('Based on the demographic profile above', 'Based on the information above')}"""
    
    return prompt


def create_vaccine_prompt_with_context(
    demographic_description: str,
    dictator_answer: Optional[int] = None,
    ultimatum_answer: Optional[int] = None
) -> str:
    """
    Create a prompt for vaccine prediction using demographics + dictator + ultimatum answers.
    
    Args:
        demographic_description: Natural language description of demographics
        dictator_answer: Dictator game answer (1-6) or None
        ultimatum_answer: Ultimatum game sender answer (1-6) or None
        
    Returns:
        Complete prompt string
    """
    context_parts = [f"## Demographic Profile:\n{demographic_description}"]
    
    # Add dictator game answer if provided
    if dictator_answer is not None:
        dictator_text = get_dictator_question_text().split("## Instructions:")[0].strip()
        dictator_answer_text = format_dictator_answer(dictator_answer)
        context_parts.append(f"## Dictator Game:\n{dictator_text}\n\nThis person chose: Option {dictator_answer} - {dictator_answer_text}")
    
    # Add ultimatum game answer if provided
    if ultimatum_answer is not None:
        ultimatum_text = get_ultimatum_sender_question_text().split("## Instructions:")[0].strip()
        ultimatum_answer_text = format_ultimatum_answer(ultimatum_answer)
        context_parts.append(f"## Ultimatum Game (Sender):\n{ultimatum_text}\n\nThis person chose: Option {ultimatum_answer} - {ultimatum_answer_text}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""{context}

---

## Question to Predict:
{get_vaccine_question_text().replace('Based on the demographic profile above', 'Based on the information above')}"""
    
    return prompt


# Keep old functions for backward compatibility (if needed)
def create_vaccine_prompt(demographic_description: str) -> str:
    """Create a prompt for vaccine prediction (demographics only)."""
    return create_vaccine_prompt_with_context(demographic_description, None, None)


def create_trust_sender_prompt(demographic_description: str) -> str:
    """Create a prompt for Trust game sender prediction (demographics only)."""
    return create_trust_sender_prompt_with_context(demographic_description, None, None)


def parse_numeric_response(response_text: str, min_val: int = 1, max_val: int = 6) -> Optional[int]:
    """
    Parse numeric prediction from LLM response.
    
    Looks for a number between min_val and max_val in the response text.
    
    Args:
        response_text: Raw response text from LLM
        min_val: Minimum valid value (default: 1)
        max_val: Maximum valid value (default: 6)
        
    Returns:
        Parsed number or None if not found/invalid
    """
    if not response_text:
        return None
    
    # Clean the response
    response_text = response_text.strip()
    
    # Try to find a number in the valid range
    # Look for standalone numbers first
    numbers = re.findall(rf'\b([{min_val}-{max_val}])\b', response_text)
    if numbers:
        val = int(numbers[0])
        if min_val <= val <= max_val:
            return val
    
    # Look for any numbers and check if in range
    all_numbers = re.findall(r'\b(\d+)\b', response_text)
    for num_str in all_numbers:
        val = int(num_str)
        if min_val <= val <= max_val:
            return val
    
    # Try to extract from common patterns
    patterns = [
        rf'answer[:\s]+([{min_val}-{max_val}])',
        rf'prediction[:\s]+([{min_val}-{max_val}])',
        rf'response[:\s]+([{min_val}-{max_val}])',
        rf'option[:\s]+([{min_val}-{max_val}])',
        rf'([{min_val}-{max_val}])[\.\)]',  # Number followed by period or parenthesis
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if min_val <= val <= max_val:
                return val
    
    return None


def parse_vaccine_response(response_text: str) -> Optional[int]:
    """
    Parse vaccine likelihood prediction from LLM response (1-4).
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Parsed number (1-4) or None if not found/invalid
    """
    return parse_numeric_response(response_text, min_val=1, max_val=4)


def parse_game_response(response_text: str) -> Optional[int]:
    """
    Parse game prediction from LLM response (1-6 for most games).
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Parsed number (1-6) or None if not found/invalid
    """
    return parse_numeric_response(response_text, min_val=1, max_val=6)


def validate_vaccine_response(response_text: str) -> bool:
    """
    Validate that the response contains a valid vaccine prediction (1-4).
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        True if valid, False otherwise
    """
    parsed = parse_vaccine_response(response_text)
    return parsed is not None and 1 <= parsed <= 4


def validate_game_response(response_text: str, min_val: int = 1, max_val: int = 6) -> bool:
    """
    Validate that the response contains a valid game prediction.
    
    Args:
        response_text: Raw response text from LLM
        min_val: Minimum valid value (default: 1)
        max_val: Maximum valid value (default: 6)
        
    Returns:
        True if valid, False otherwise
    """
    parsed = parse_numeric_response(response_text, min_val, max_val)
    return parsed is not None and min_val <= parsed <= max_val


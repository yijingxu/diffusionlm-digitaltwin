"""
Helper functions for vaccine prediction task.

These functions support the simplified demographic-only vaccine prediction workflow.
"""

import json
import re
from typing import Dict, Optional, Any


def get_vaccine_question_text() -> str:
    """
    Get the vaccine question (QID291) text and format instructions.
    
    This is a static question, so we can hardcode it or extract from one example file.
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


def create_vaccine_prompt(demographic_description: str) -> str:
    """
    Create a prompt for vaccine prediction using demographic description.
    
    Args:
        demographic_description: Natural language description of demographics
        
    Returns:
        Complete prompt string
    """
    prompt = f"""## Demographic Profile:
{demographic_description}

---

## Question:
{get_vaccine_question_text()}"""
    
    return prompt


def parse_vaccine_response(response_text: str) -> Optional[int]:
    """
    Parse vaccine likelihood prediction from LLM response.
    
    Looks for a number between 1-4 in the response text.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Parsed number (1-4) or None if not found/invalid
    """
    if not response_text:
        return None
    
    # Clean the response
    response_text = response_text.strip()
    
    # Try to find a number 1-4
    # Look for standalone numbers first
    numbers = re.findall(r'\b([1-4])\b', response_text)
    if numbers:
        return int(numbers[0])
    
    # Look for numbers that might be part of a sentence
    numbers = re.findall(r'([1-4])', response_text)
    if numbers:
        return int(numbers[0])
    
    # Try to extract from common patterns
    patterns = [
        r'answer[:\s]+([1-4])',
        r'prediction[:\s]+([1-4])',
        r'likelihood[:\s]+([1-4])',
        r'response[:\s]+([1-4])',
        r'([1-4])[\.\)]',  # Number followed by period or parenthesis
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None


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


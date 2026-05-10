#!/usr/bin/env python3
"""
Generate prompts by replacing {QUESTION} placeholder in template with each question from questions.txt
"""

import json
import re


def load_template(template_path: str) -> str:
    """Load template file as a string."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_questions(questions_path: str) -> list:
    """Load questions from file, skipping empty lines."""
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


def generate_prompts(template_path: str, questions_path: str, output_path: str, placeholder: str = "{QUESTION}"):
    """
    Generate prompts by replacing placeholder in template with each question.
    Saves prompts as both JSON and text file formats.
    
    Args:
        template_path: Path to template file containing {QUESTION} placeholder
        questions_path: Path to questions file (one per line)
        output_path: Path to save JSON output
        placeholder: Placeholder string to replace (default: {QUESTION})
    """
    # Load template and questions
    template = load_template(template_path)
    questions = load_questions(questions_path)
    
    # Verify placeholder exists in template
    if placeholder not in template:
        raise ValueError(
            f"Placeholder '{placeholder}' not found in template file '{template_path}'. "
            f"Please add '{placeholder}' where you want questions to be inserted."
        )
    
    print(f"Loaded template ({len(template)} characters)")
    print(f"Loaded {len(questions)} questions")
    print(f"Using placeholder: {placeholder}")
    
    # Generate prompts
    prompts = []
    for question in questions:
        # Replace placeholder with question
        prompt = template.replace(placeholder, question)
        prompts.append(prompt)
    
    # Save as JSON (for infer_vllm.py)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(prompts)} prompts")
    print(f"Saved prompts to {output_path}")
    
    # Also save as text file (one prompt per line, but prompts contain newlines)
    # For text format, we'll save with a separator
    text_output_path = output_path.replace('.json', '.txt')
    with open(text_output_path, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"=== PROMPT {i+1} ===\n")
            f.write(prompt)
            f.write("\n\n")
    
    print(f"Also saved text version to {text_output_path}")
    
    return prompts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate prompts from template and questions"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="evil_template.txt",
        help="Path to template file (default: template.txt)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="evil_questions.txt",
        help="Path to questions file (default: questions.txt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../outputs/evil_generated_prompts.json",
        help="Output JSON file path (default: ../outputs/generated_prompts.json)"
    )
    parser.add_argument(
        "--placeholder",
        type=str,
        default="{QUESTION}",
        help="Placeholder string to replace in template (default: {QUESTION})"
    )
    
    args = parser.parse_args()
    
    generate_prompts(args.template, args.questions, args.output, args.placeholder)


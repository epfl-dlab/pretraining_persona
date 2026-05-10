# This file generates character description extraction and evaluation data for a specified trait using Google Gemini models.
# It transforms generic instructions and questions into character-specific versions for a fictional character named 'Alex'.

import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


def main(trait = "evil", output_file_name: str = "evil_character_neutral_q.json", model: str = "gemini-2.5-pro", temperature: float = 0.0):
    # load extract and eval json from /data/{trait}.json as json
    folders = ["trait_data_eval", "trait_data_extract"]
    instructions = []
    questions = []
    eval_prompt = ""
    for folder in folders:
        with open(f"data/{folder}/{trait}.json", "r") as file:
            data = json.load(file)
        if not len(instructions):
            instructions = data['instruction']
        else:
            assert instructions == data['instruction'], "Instructions do not match!"

        questions.append(data['questions'])

        if not len(eval_prompt):
            eval_prompt = data['eval_prompt']
        else:
            assert eval_prompt == data['eval_prompt'], "Eval prompts do not match!"
    
    print("Instructions: {}".format(instructions))
    print("Questions: {}".format(questions))
    print(eval_prompt)
    
    load_dotenv()
    client = genai.Client() # requires GEMINI_API_KEY in environment

    # instructions
    prompt = f"Transform the following instructions one by one into a character portrayal for a fictional person called 'Alex'. Keep the pos instructions {trait} and the neg instruction not {trait}. Make them readible stand-alone, do not refer to other instructions. Output the same JSON structure.\n\n{instructions}"

    response = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )

    instructions = json.loads(response.text.replace("```json", "").replace("```", ""))
    print(instructions)
    time.sleep(30)

    prompt = f"Transform the following questions one by one into a hypothetical act by a person called 'Alex', e.g. 'If Alex had to'. Change as little as possible but enable meaningful continuation of the sentence. Make them readible stand-alone, do not refer to other questions. Output the same array structure.\n\n{questions[0] + questions[1]}"

    response = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )
    print(response.text)
    
    
    situations = json.loads(response.text.replace("```json", "").replace("```", "").replace("...", ""))
    print(situations)
    extract_questions = situations[:len(questions[0])]
    eval_questions = situations[len(questions[0]):]
    time.sleep(30)

    # eval prompt
    prompt = "Transform the following question-response evaluation prompt into a hypothetical situation continuation evaluation. Keep the placeholders {question} and {answer} intact.\n\n"+eval_prompt
    response = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )
    eval_prompt = response.text


    print(eval_prompt) 

    for folder, questions in zip(folders, [extract_questions, eval_questions]):
        output = {
            "instruction": instructions,
            "questions": questions,
            "eval_prompt": eval_prompt
        }
        with open(f"data/{folder}/{output_file_name}", "w") as file:
            pass

        print(f"Saved character descriptions to {folder}/{output_file_name}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
    
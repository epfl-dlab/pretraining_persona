import os
import asyncio
import yaml
from typing import Dict, List, Optional
import json
from pathlib import Path
import torch
import pandas as pd
import random
from datasets import load_dataset
from tqdm import tqdm

import torch
try:
    from vllm import LLM as VLLMEngine, SamplingParams
    from vllm.lora.request import LoRARequest
    _VLLM_AVAILABLE = True
except ImportError:
    VLLMEngine = None
    SamplingParams = None
    LoRARequest = None
    _VLLM_AVAILABLE = False

from itertools import islice
import asyncio
from source.activation_steer import ActivationSteerer
from source.judge import _sanitize_openai_text
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from source.model_utils import load_model, load_vllm_model
import logging
from source.prompts import Prompts

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)


def _judge_class_for_model(model_name: str):
    provider = os.environ.get("JUDGE_PROVIDER", "").strip().lower()
    model_lower = str(model_name).strip().lower()
    if provider == "deepseek" or (not provider and model_lower.startswith("deepseek")):
        from source.deepseek_judge import DeepSeekJudge

        return DeepSeekJudge
    if provider not in {"", "openai"}:
        raise ValueError(
            f"Unsupported JUDGE_PROVIDER={provider!r}. "
            "Supported providers: openai, deepseek."
        )
    from source.judge import OpenAiJudge

    return OpenAiJudge


def get_text(messages, tokenizer):
    """Convert messages to text using the tokenizer's chat template if available."""
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return " ".join([m['content'] for m in messages])

def sample_steering(model, tokenizer, conversations,  vector, layer, coef, bs=20, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, steering_type="response", repetition_penalty=1):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for messages in conversations:
        prompts.append(get_text(messages, tokenizer))
    
    outputs = []
    for i in trange(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer-1, positions=steering_type):
            with torch.no_grad():
                output = model.generate(**tokenized_batch, do_sample=(temperature > 0), temperature=temperature, top_p=top_p, max_new_tokens=max_tokens,use_cache=True, min_new_tokens=min_tokens, repetition_penalty=repetition_penalty)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)
    return prompts, outputs


def sample(model, tokenizer, conversations, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, lora_path=None, repetition_penalty=1, batch_size=20):
    texts = []
    for i, messages in enumerate(conversations):
        texts.append(get_text(messages, tokenizer))


    is_vllm = _VLLM_AVAILABLE and isinstance(model, VLLMEngine)
    if is_vllm:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            skip_special_tokens=True,
            stop=[tokenizer.eos_token],
            min_tokens=min_tokens,
            repetition_penalty=repetition_penalty
        )
        
        generate_kwargs = {
            "sampling_params": sampling_params,
            "use_tqdm": True
        }
        if lora_path is not None:
            completions = model.generate(texts, **generate_kwargs, lora_request=LoRARequest("default", 1, lora_path=lora_path))
        else:
            completions = model.generate(texts, **generate_kwargs)
        answers = [completion.outputs[0].text for completion in completions]
        return texts, answers
    else:
        decoded = []
        for i in trange(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            input_batch = tokenizer(batch_texts, return_tensors="pt", padding=True)
            input_batch = {k: v.to(model.device) for k, v in input_batch.items()}

            with torch.no_grad():
                out = model.generate(
                    **input_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    min_new_tokens=min_tokens,
                    repetition_penalty=repetition_penalty,
                    use_cache=True,
                )

            prompt_lengths = input_batch["attention_mask"].sum(dim=1).tolist()
            decoded.extend(
                tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
                for o, prompt_len in zip(out, prompt_lengths)
            )
        return texts, decoded



def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def save_outputs(df: pd.DataFrame, output_path: str, message: Optional[str] = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    if message:
        print(f"{message}: {output_path}")
    else:
        print(output_path)


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o",
            judge_eval_type: str = "0_100",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judge_model = judge
        self.judge_eval_type = judge_eval_type
        self.judge_prompts = judge_prompts
        self.judges = None

    def _ensure_judges(self):
        if self.judges is None:
            JudgeClass = _judge_class_for_model(self.judge_model)

            self.judges = {
                metric: JudgeClass(
                    self.judge_model,
                    prompt,
                    eval_type=self.judge_eval_type if metric != "coherence" else "0_100",
                )
                for metric, prompt in self.judge_prompts.items()
            }
    
    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role='user', content=i)] for i in paraphrases]
        if self.system:
            conversations = [[dict(role='system', content=self.system)] + c for c in conversations]
        return paraphrases, conversations

    def generate_df(self, llm, tokenizer, coef, vector=None, layer=None, max_tokens=1000, n_per_question=100, steering_type="last", lora_path=None, repetition_penalty=1):
        paraphrases, conversations = self.get_input(n_per_question)
        if coef != 0:
            prompts, answers = sample_steering(llm, tokenizer, conversations, vector, layer, coef, temperature=self.temperature, max_tokens=max_tokens, steering_type=steering_type, repetition_penalty=repetition_penalty)
        else:
            prompts, answers = sample(llm, tokenizer, conversations, temperature=self.temperature, max_tokens=max_tokens, lora_path=lora_path, repetition_penalty=repetition_penalty)
        return pd.DataFrame([
            dict(question=question,prompt=prompt, answer=answer, question_id=self.id)
            for question, answer, prompt in zip(paraphrases, answers, prompts)
        ])

    async def add_judgments(self, df: pd.DataFrame):
        self._ensure_judges()
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(df["question"], df["answer"])
            ])
            df[score] = scores
        return df

    async def eval(self, llm, tokenizer, coef, vector=None, layer=None, max_tokens=1000, n_per_question=100, steering_type="last", lora_path=None, skip_judge=False, repetition_penalty=1):
        df = self.generate_df(llm, tokenizer, coef, vector, layer, max_tokens, n_per_question, steering_type, lora_path, repetition_penalty)
        if skip_judge:
            return df
        return await self.add_judgments(df)
        

def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"

def _trait_data_path(trait: str, version: str, trait_data_dir: Optional[str] = None) -> Path:
    base_dir = Path(trait_data_dir) if trait_data_dir is not None else Path(f"data/trait_data_{version}")
    return base_dir / f"{trait}.json"


def load_persona_questions(trait, temperature=1, persona_instructions_type=None, assistant_name=None, judge_model="gpt-4.1-mini-2025-04-14", eval_type="0_100", version="eval", address_assistant=True, few_shot=None, max_questions: Optional[int] = None, trait_data_dir: Optional[str] = None):
    with open(_trait_data_path(trait, version, trait_data_dir), "r") as handle:
        trait_data = json.load(handle)
    judge_prompts = {}
    prompt_template = trait_data["eval_prompt"]
    judge_prompts[trait] = prompt_template
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]
    raw_questions = trait_data["questions"]
    if max_questions is not None and max_questions > 0:
        raw_questions = raw_questions[:max_questions]
    questions = []
    for i, question in enumerate(raw_questions):
        if persona_instructions_type is not None:
            persona_instructions = [x[persona_instructions_type] for x in trait_data["instruction"]]
            for k, instruction in enumerate(persona_instructions):
                # Handle instructions as lists (arrays) for few-shot examples
                if isinstance(instruction, list):
                    # If few_shot is specified, use first N examples
                    if few_shot is not None:
                        instruction_parts = instruction[:few_shot]
                    else:
                        instruction_parts = instruction
                    # Join list elements with newlines to create few-shot examples
                    instruction = "\n".join(instruction_parts)
                
                if assistant_name is None:
                    if persona_instructions_type == "pos":
                        assistant_name = trait
                    else:
                        assistant_name = "helpful"
                system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}" if address_assistant else instruction
                questions.append(Question(paraphrases=[question], id=f"{trait}_{i}_{persona_instructions_type}_{k}", judge_prompts=judge_prompts, judge=judge_model, temperature=temperature, system=system, judge_eval_type=eval_type ))
        else:
            questions.append(Question(paraphrases=[question], id=f"{trait}_{i}", judge_prompts=judge_prompts, judge=judge_model, temperature=temperature, judge_eval_type=eval_type ))
    return questions





async def eval_batched(questions, llm, tokenizer, coef, vector=None, layer=None, n_per_question=100, max_concurrent_judges=100, max_tokens=1000, steering_type="last", lora_path=None, repetition_penalty=1, skip_judge=False, generation_batch_size=20, pre_judge_output_path: Optional[str] = None):
    """Batch process all questions together for faster inference"""
    # Collect all prompts from all questions
    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))
    
    # Generate all answers in a single batch
    print(f"Generating {len(all_conversations)} responses in a single batch...")
    if coef != 0:
        prompts, answers = sample_steering(llm, tokenizer, all_conversations, vector, layer, coef, bs=generation_batch_size, temperature=questions[0].temperature, max_tokens=max_tokens, steering_type=steering_type, repetition_penalty=repetition_penalty)
    else:
        prompts, answers = sample(llm, tokenizer, all_conversations, temperature=questions[0].temperature, max_tokens=max_tokens, lora_path=lora_path, repetition_penalty=repetition_penalty, batch_size=generation_batch_size)
    
    # Prepare data structures for batch evaluation
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []  # Store (question_idx, metric, sample_idx) for each task
    
    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        # Get this question's data
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]
        
        # Create dataframe for this question
        df = pd.DataFrame([
            dict(question=question_text, prompt=prompt, answer=answer, question_id=question.id)
            for question_text, answer, prompt in zip(q_paraphrases, q_answers, q_prompts)
        ])
        question_dfs.append(df)

        if skip_judge:
            continue

        question._ensure_judges()
        # Collect all judge tasks
        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(zip(q_paraphrases, q_answers)):
                all_judge_tasks.append((judge, question_text, answer))
                all_judge_indices.append((i, metric, sample_idx))

    if skip_judge:
        return question_dfs

    if pre_judge_output_path is not None:
        save_outputs(pd.concat(question_dfs), pre_judge_output_path, "Saved raw generations before judging")

    # Run judge evaluations with concurrency control
    print(f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests...")
    all_results = [None] * len(all_judge_tasks)  # Pre-allocate results array
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent_judges)
    
    async def run_with_semaphore(task_idx, judge, question_text, answer, metric, question_id, sample_idx):
        async with semaphore:
            try:
                result = await judge(question=question_text, answer=answer)
            except Exception as exc:
                answer_preview = _sanitize_openai_text(str(answer), max_len=160).replace("\n", " ")
                raise RuntimeError(
                    f"Judge request failed metric={metric} question_id={question_id} "
                    f"sample_idx={sample_idx} answer_preview={answer_preview!r}"
                ) from exc
            return task_idx, result
    
    # Create all tasks with semaphore control
    tasks = [
        run_with_semaphore(
            task_idx,
            judge,
            question_text,
            answer,
            all_judge_indices[task_idx][1],
            questions[all_judge_indices[task_idx][0]].id,
            all_judge_indices[task_idx][2],
        )
        for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)
    ]
    
    # Process tasks in batches with progress bar
    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result  # Store result in correct position
            pbar.update(1)
    
    # Distribute results back to the appropriate dataframes
    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result
    
    return question_dfs


async def judge_saved_outputs(df: pd.DataFrame, questions, max_concurrent_judges: int):
    question_by_id = {question.id: question for question in questions}
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []

    for question_idx, (question_id, question_df) in enumerate(df.groupby("question_id", sort=False)):
        question = question_by_id.get(question_id)
        if question is None:
            raise ValueError(f"Saved generations contain unknown question_id: {question_id}")
        question._ensure_judges()
        question_df = question_df.reset_index(drop=True).copy()
        question_dfs.append(question_df)
        for metric, judge in question.judges.items():
            for sample_idx, row in question_df.iterrows():
                all_judge_tasks.append((judge, row["question"], row["answer"]))
                all_judge_indices.append((question_idx, metric, sample_idx))

    print(f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests on saved generations...")
    all_results = [None] * len(all_judge_tasks)
    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_with_semaphore(task_idx, judge, question_text, answer, metric, question_id, sample_idx):
        async with semaphore:
            try:
                result = await judge(question=question_text, answer=answer)
            except Exception as exc:
                answer_preview = _sanitize_openai_text(str(answer), max_len=160).replace("\n", " ")
                raise RuntimeError(
                    f"Judge request failed metric={metric} question_id={question_id} "
                    f"sample_idx={sample_idx} answer_preview={answer_preview!r}"
                ) from exc
            return task_idx, result

    tasks = [
        run_with_semaphore(
            task_idx,
            judge,
            question_text,
            answer,
            all_judge_indices[task_idx][1],
            question_dfs[all_judge_indices[task_idx][0]].loc[all_judge_indices[task_idx][2], "question_id"],
            all_judge_indices[task_idx][2],
        )
        for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)
    ]

    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result
            pbar.update(1)

    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return pd.concat(question_dfs, ignore_index=True)

def main(model, trait, output_path, coef=0, vector_path=None, layer=None, steering_type="response", max_tokens=1000, n_per_question=10, batch_process=True, max_concurrent_judges=100, persona_instruction_type=None, assistant_name=None, judge_model="gpt-4.1-mini-2025-04-14", version="extract", overwrite=False, skip_judge=False, repetition_penalty=1, few_shot=None, temperature: Optional[float] = None, revision: Optional[str] = None, vector_norm: float = None, source_activation_norm: float = None, target_activation_norm: float = None, prefer_transformers: bool = False, max_questions: Optional[int] = None, generation_batch_size: int = 20, trait_data_dir: Optional[str] = None):
    """Evaluate a model on all questions form the evaluation yaml file"""
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        if not skip_judge and not all(metric in df.columns for metric in [trait, "coherence"]):
            print("Saved generations found without judge columns; resuming judging from disk.")
            resume_tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
            is_instruct_model = hasattr(resume_tokenizer, "chat_template") and resume_tokenizer.chat_template is not None
            questions = load_persona_questions(
                trait,
                temperature=0.0,
                persona_instructions_type=persona_instruction_type,
                assistant_name=assistant_name,
                judge_model=judge_model,
                version=version,
                address_assistant=is_instruct_model,
                few_shot=few_shot,
                eval_type="0_100" if is_instruct_model else "base_0_100",
                max_questions=max_questions,
                trait_data_dir=trait_data_dir,
            )
            outputs = asyncio.run(judge_saved_outputs(df, questions, max_concurrent_judges))
            save_outputs(outputs, output_path)
            for metric in [trait, "coherence"]:
                if metric in outputs.columns:
                    print(f"{metric}:  {outputs[metric].mean():.2f} +- {outputs[metric].std():.2f}")
            return
        for metric in [trait, "coherence"]:
            if metric in df.columns:
                print(f"{metric}:  {df[metric].mean():.2f} +- {df[metric].std():.2f}")
        return
    
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Decide what temperature to use for generation
    if temperature is None:
        # Keep the old behavior: 0.0 if single sample, 1.0 otherwise
        if n_per_question == 1:
            gen_temperature = 0.0
        else:
            gen_temperature = 0.5
    else:
        # User explicitly provided Temperature (0.4 for the last generation)
        gen_temperature = temperature

    use_transformers = coef != 0 or prefer_transformers or not _VLLM_AVAILABLE
    if use_transformers:
        if coef == 0 and not prefer_transformers and not _VLLM_AVAILABLE:
            print("vLLM is unavailable; falling back to transformers generation.")
        llm, tokenizer = load_model(model, revision=revision)
        lora_path = None
        if coef != 0 and vector_path is None:
            raise ValueError("coef != 0 requires vector_path")
        if coef != 0 and layer is None:
            raise ValueError("coef != 0 requires layer")
        vector = torch.load(vector_path, weights_only=False)[layer] if (coef != 0 and vector_path is not None) else None
        if vector is not None and source_activation_norm is not None and target_activation_norm is not None:
            source_activation_norm = float(source_activation_norm)
            target_activation_norm = float(target_activation_norm)
            if source_activation_norm == 0.0:
                raise ValueError("source_activation_norm must be non-zero")
            transfer_scale = target_activation_norm / source_activation_norm
            vector = vector * transfer_scale
            print(
                "Vector transfer scaled:",
                transfer_scale,
                "source_norm=",
                source_activation_norm,
                "target_norm=",
                target_activation_norm,
                "resulting_vector_norm=",
                torch.norm(vector).item(),
            )
        elif vector is not None and vector_norm is not None:
            vector = vector / torch.norm(vector) * vector_norm
            print("Vector scaled to norm:", torch.norm(vector).item())
    else:
        llm, tokenizer, lora_path = load_vllm_model(model, revision=revision)
        vector=None
    is_instruct_model = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    print("0_100" if is_instruct_model else "base_0_100")
    questions = load_persona_questions(trait, temperature=gen_temperature, persona_instructions_type=persona_instruction_type, assistant_name=assistant_name, judge_model=judge_model, version=version, address_assistant=is_instruct_model, few_shot=few_shot, eval_type="0_100" if is_instruct_model else "base_0_100", max_questions=max_questions, trait_data_dir=trait_data_dir)

    if batch_process:
        print(f"Batch processing {len(questions)} '{trait}' questions...")
        outputs_list = asyncio.run(eval_batched(questions, llm, tokenizer,coef, vector, layer, n_per_question, max_concurrent_judges, max_tokens, steering_type=steering_type, lora_path=lora_path, repetition_penalty=repetition_penalty, skip_judge=skip_judge, generation_batch_size=generation_batch_size, pre_judge_output_path=output_path if not skip_judge else None))
        outputs = pd.concat(outputs_list)
    else:
        generated_outputs = []
        for question in tqdm(questions, desc=f"Processing {trait} questions"):
            generated_outputs.append(question.generate_df(llm, tokenizer,coef, vector, layer, max_tokens, n_per_question, steering_type=steering_type, lora_path=lora_path, repetition_penalty=repetition_penalty))
        generated = pd.concat(generated_outputs)
        save_outputs(generated, output_path, "Saved raw generations before judging" if not skip_judge else "Saved generations")
        if skip_judge:
            outputs = generated
        else:
            judged_outputs = []
            for question, generated_df in zip(questions, generated_outputs):
                judged_outputs.append(asyncio.run(question.add_judgments(generated_df.copy())))
            outputs = pd.concat(judged_outputs)
    save_outputs(outputs, output_path)
    if not skip_judge:
        for trait in [trait , "coherence"]:
            print(f"{trait}:  {outputs[trait].mean():.2f} +- {outputs[trait].std():.2f}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)

#!/usr/bin/env python3
"""
Script for inferring a model with vLLM and saving responses to JSON.
Supports single prompts, multiple prompts from file, or interactive mode.
"""

import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a JSON or text file."""
    with open(file_path, 'r') as f:
        if file_path.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'prompts' in data:
                return data['prompts']
            else:
                raise ValueError("JSON file must contain a list of prompts or a dict with 'prompts' key")
        else:
            # Assume text file with one prompt per line
            return [line.strip() for line in f if line.strip()]


def save_responses_to_json(responses: List[Dict[str, Any]], output_path: str):
    """Save responses to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    print(f"Responses saved to {output_path}")


def infer_with_vllm(
    model_name: str,
    prompts: List[str],
    output_path: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    **kwargs
):
    """
    Run inference with vLLM and save results to JSON.
    
    Args:
        model_name: HuggingFace model name or path
        prompts: List of prompts to generate responses for
        output_path: Path to save JSON output
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization ratio
        **kwargs: Additional vLLM LLM parameters
    """
    print(f"Loading model: {model_name}")
    print(f"Number of prompts: {len(prompts)}")
    
    # Initialize vLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Format responses
    responses = []
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        generated_text = output.outputs[0].text
        response = {
            "prompt": prompt,
            "response": generated_text,
            "prompt_id": i,
            "finish_reason": output.outputs[0].finish_reason,
            "token_count": len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else None
        }
        responses.append(response)
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print(f"Response: {generated_text[:200]}..." if len(generated_text) > 200 else f"Response: {generated_text}")
    
    # Save to JSON
    save_responses_to_json(responses, output_path)
    
    return responses


def interactive_mode(
    model_name: str,
    output_path: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    **kwargs
):
    """Interactive mode for asking questions one by one."""
    print(f"Loading model: {model_name}")
    
    # Initialize vLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    responses = []
    prompt_id = 0
    
    print("\n=== Interactive Mode ===")
    print("Enter your prompts (type 'quit' or 'exit' to finish, 'save' to save and continue)")
    print("-" * 50)
    
    while True:
        prompt = input(f"\n[Prompt {prompt_id + 1}] Enter your question: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        if not prompt:
            continue
        
        # Generate response
        print("Generating response...")
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        response = {
            "prompt": prompt,
            "response": generated_text,
            "prompt_id": prompt_id,
            "finish_reason": outputs[0].outputs[0].finish_reason
        }
        responses.append(response)
        
        print(f"\nResponse:\n{generated_text}")
        print("-" * 50)
        
        prompt_id += 1
        
        # Ask if user wants to save
        save_now = input("Save current responses? (y/n, default: n): ").strip().lower()
        if save_now == 'y':
            save_responses_to_json(responses, output_path)
    
    # Save all responses at the end
    if responses:
        save_responses_to_json(responses, output_path)
        print(f"\nTotal prompts processed: {len(responses)}")
    else:
        print("\nNo prompts were processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Infer a model with vLLM and save responses to JSON"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf",
        help="Model name or path (default: DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to use"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to file containing prompts (JSON list or text file with one prompt per line)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (ask questions one by one)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vllm_responses.json",
        help="Output JSON file path (default: vllm_responses.json)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9)"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.interactive:
        interactive_mode(
            model_name=args.model,
            output_path=args.output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.prompt:
        prompts = [args.prompt]
        infer_with_vllm(
            model_name=args.model,
            prompts=prompts,
            output_path=args.output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    elif args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        infer_with_vllm(
            model_name=args.model,
            prompts=prompts,
            output_path=args.output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    else:
        parser.error("Must specify either --prompt, --prompts-file, or --interactive")


if __name__ == "__main__":
    main()


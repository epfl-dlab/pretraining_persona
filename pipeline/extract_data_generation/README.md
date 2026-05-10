# Generating stories for persona vector extraction

This guide shows how to generate prompts from the template and questions, then run inference with vLLM.

## Files

1. **`generate_prompts.py`** - Script that replaces `{QUESTION}` placeholder in `template.txt` with each question from `questions.txt`
2. **`infer_vllm.py`** - Main inference script using vLLM
3. **`run_inference.sh`** - Convenience script to run inference
4. **`template.txt`** - Template file with `{QUESTION}` placeholder
5. **`questions.txt`** - List of questions (one per line)

## Template Format

The `template.txt` file uses a `{QUESTION}` placeholder that will be replaced with each question. You can place `{QUESTION}` anywhere in the template, making it flexible and robust.

## Step 1: Generate Prompts (if needed)

From the `inference/` directory, regenerate the prompts:

```bash
cd inference
python3 generate_prompts.py
```

This will create `../outputs/generated_prompts.json` and `../outputs/generated_prompts.txt`.

Or with custom paths and placeholder:

```bash
python3 generate_prompts.py \
    --template template.txt \
    --questions questions.txt \
    --output ../outputs/generated_prompts.json \
    --placeholder "{QUESTION}"
```

## Step 2: Run Inference with vLLM

### Option A: Using the convenience script

From the `inference/` directory:

```bash
cd inference
./run_inference.sh
```

### Option B: Using the Python script directly

From the `inference/` directory:

```bash
cd inference
python3 infer_vllm.py \
    --model "DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf" \
    --prompts-file ../outputs/generated_prompts.json \
    --output ../outputs/vllm_responses.json \
    --max-tokens 512 \
    --temperature 0.7 \
    --top-p 0.9 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

### Option C: Customize parameters via environment variables

```bash
cd inference
export MODEL="your-model-name"
export MAX_TOKENS=1024
export TEMPERATURE=0.8
./run_inference.sh
```

## Parameters

- `--model`: Model name or path (default: DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf)
- `--prompts-file`: Path to JSON file with prompts (default: ../outputs/generated_prompts.json)
- `--output`: Output JSON file path (default: ../outputs/vllm_responses.json)
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: 1)
- `--gpu-memory-utilization`: GPU memory utilization ratio (default: 0.9)

## Output

The inference results will be saved to `../outputs/vllm_responses.json` (or your specified output file) with the following structure:

```json
[
  {
    "prompt": "...",
    "response": "...",
    "prompt_id": 0,
    "finish_reason": "length",
    "token_count": 512
  },
  ...
]
```


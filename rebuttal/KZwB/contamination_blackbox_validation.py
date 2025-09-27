import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
from typing import List, Tuple
from datasets import load_dataset

import scipy
from openai import OpenAI
import os
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
download_dir = ""

import argparse
import numpy as np

def get_token_logprobs(model, tokenizer, text: str) -> Tuple[List[float], List[str]]:
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"]
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    gather_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return gather_log_probs[0].tolist(), tokenizer.convert_ids_to_tokens(shift_labels[0])

def get_token_logprobs_openai(model_name: str, text: str) -> Tuple[List[float], List[str]]:
    client = OpenAI(api_key="")
    messages = [
        {"role": "user", "content": text}
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=50,
        logprobs=True,
        top_logprobs=20
    )
   
    logprob_items = response.choices[0].logprobs.content
    tokens = [item.token for item in logprob_items]
    token_logprobs = [item.logprob for item in logprob_items]

    filtered_pairs = [(t, lp) for t, lp in zip(tokens, token_logprobs) if lp is not None]
    if not filtered_pairs:
        raise ValueError("No valid logprobs returned; check model availability.")
    tokens_clean, lps_clean = zip(*filtered_pairs)
    return list(lps_clean), list(tokens_clean)



def min_k_percent(log_probs: List[float], k: float = 0.1) -> float:
    n = len(log_probs)
    assert 0 < k <= 1, "k must be in (0,1]"
    m = max(1, int(math.ceil(n * k)))
    return float(np.mean(np.sort(log_probs)[:m]))

def pdd_score(log_probs: List[float]) -> float:
    probs = np.exp(log_probs)
    probs = probs / probs.sum()
    uniform = np.full_like(probs, 1.0 / len(probs))
    m = 0.5 * (probs + uniform)
    js = 0.5 * (scipy.stats.entropy(probs, m) + scipy.stats.entropy(uniform, m))
    return js

def classify_contamination(min_k_score: float, threshold: float = -2.0) -> bool:
    return min_k_score > threshold  # less negative -> higher probability

def main():
    parser = argparse.ArgumentParser(description="Min-k% black-box contamination test")
    parser.add_argument("--model", default="gpt-4o", help="HF repo or local model dir")
    parser.add_argument("--file_path", default="", help="Sentence file to test")
    parser.add_argument("--k", type=float, default=0.05, help="Lower-tail fraction (0<k≤1)")
    parser.add_argument("--threshold", type=float, default=-2.0, help="Decision threshold")

    args = parser.parse_args()

    if args.model.lower().startswith("gpt-") is False:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, cache_dir=download_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            cache_dir=download_dir
        )
        model.eval()

    dataset = load_dataset(
        "json",
        data_files="[data folder]/*.jsonl", 
        split="train"               
    )
    detected_count = 0
    for item in dataset:
        scenario = item["scenario"]
        if args.model.lower().startswith("gpt-"):
            log_probs, tokens = get_token_logprobs_openai(args.model, scenario)
        else:
            log_probs, tokens = get_token_logprobs(model, tokenizer, scenario)

        # Display per-token log-probs
        rows = [f"{tok:>15} : {lp:.3f}" for tok, lp in zip(tokens, log_probs)]
        print("Per-token log-probs (shifted):\n" + "\n".join(rows))

        # Min-k% statistic
        min_k = min_k_percent(log_probs, args.k)
        contam = classify_contamination(min_k, args.threshold)

        print(f"\nMin-{int(args.k*100)}% avg log-prob = {min_k:.3f}")
        print(f"Decision threshold           = {args.threshold:.3f}")
        print("=> Contaminated:" if contam else "=> No contamination detected.")
        if contam:
            detected_count += 1
    print(f"Detected contamination in {detected_count}/{len(dataset)} samples")
    
if __name__ == "__main__":
    main()

"""
STAGE 5

Predict sarcasm/irony for a single text or a file of texts using a fine-tuned RoBERTa model.

This script:
- normalizes text and handles emojis in one of three modes:
  T = remove, K = keep, D = demojize
- loads the selected checkpoint from models/stage3
- runs inference and returns irony probability plus emoji-based features
- supports either a single input string or a line-by-line input file via CLI

Usage:
    python predict.py --text "Great, another Monday 😒"
    python predict.py --model roberta_K --emoji_mode K --input_file MISC/test_inputs.txt
"""

import argparse
import torch
import numpy as np
import re
import unicodedata
import emosent.emosent as _emo
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ESR dict
esr_dict = {char: data["sentiment_score"]
            for char, data in _emo.EMOJI_SENTIMENT_DICT.items()}

# Preprocessing functions
def normalize_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def apply_mode_T(text):
    return "".join(ch for ch in text if ch not in esr_dict)

def apply_mode_K(text):
    return text

def apply_mode_D(text):
    result = []
    for ch in text:
        if ch in esr_dict:
            try:
                name = unicodedata.name(ch).lower().replace(" ", "_")
                result.append(f":{name}:")
            except Exception:
                result.append(ch)
        else:
            result.append(ch)
    return "".join(result)

MODE_FN = {"T": apply_mode_T, "K": apply_mode_K, "D": apply_mode_D}

# Model paths
MODEL_PATHS = {
    "roberta_T": "models/stage3/roberta_tweeteval_T",
    "roberta_K": "models/stage3/roberta_tweeteval_K",
    "roberta_D": "models/stage3/roberta_tweeteval_D",
}

# The Main Prediction function
def predict(text, model_key, emoji_mode, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = MODEL_PATHS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model = model.to(device)
    model.eval()

    # Preprocess
    mode_fn = MODE_FN[emoji_mode]
    clean_text = mode_fn(normalize_text(text))

    # Tokenize
    encoding = tokenizer(
        clean_text,
        max_length = 128, padding = "max_length", truncation = True, return_tensors = "pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    prob_irony = probs[1]
    label = "irony" if prob_irony >= threshold else "not_irony"

    # Emoji features
    emojis = [ch for ch in text if ch in esr_dict]
    esr_scores = [esr_dict[e] for e in emojis]
    esr_sentiment = round(sum(esr_scores) / len(esr_scores), 3) if esr_scores else 0.0

    return {
        "label" : label,
        "prob_irony" : round(float(prob_irony), 4), "prob_not_irony": round(float(probs[0]), 4),
        "emojis" : emojis, "esr_sentiment" : esr_sentiment, "emoji_count"   : len(emojis),
        "processed_text": clean_text,
    }

# CLI
def main():
    parser = argparse.ArgumentParser(description="Sarcasm/Irony Detection CLI")
    parser.add_argument("--text", type=str, help="Input text to classify")
    parser.add_argument("--input_file", type=str, help="Path to input file (one text per line)")
    parser.add_argument("--model",type=str, default="roberta_K",
                        choices=["roberta_T", "roberta_K", "roberta_D"],help="Model to use (default: roberta_K)")
    parser.add_argument("--emoji_mode", type=str, default="K",
                        choices=["T", "K", "D"], help="Emoji handling mode: T=remove, K=keep, D=demojize (default: K)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Classification threshold (default: 0.5)")
    args = parser.parse_args()

    # Collect texts
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Provide either --text or --input_file")

    # Run predictions
    for text in texts:
        result = predict(text, args.model, args.emoji_mode, args.threshold)
        print(f"\nText : {text}")
        print(f"Label : {result['label']}")
        print(f"Prob(irony) : {result['prob_irony']}")
        print(f"Prob(not_irony): {result['prob_not_irony']}")
        print(f"Emojis : {result['emojis']}")
        print(f"ESR sentiment : {result['esr_sentiment']}")
        print(f"Emoji count : {result['emoji_count']}")
        print(f"Processed text: {result['processed_text']}")
if __name__ == "__main__":
    main()
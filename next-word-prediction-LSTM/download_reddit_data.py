import os, re, html, json, nltk, requests
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from huggingface_hub import hf_hub_download


# nltk 준비
nltk.download("punkt")
tokenizer = TweetTokenizer()

# 전처리 함수
def preprocess(text):
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", "<URL>", text)
    text = re.sub(r"/?u/[\w-]+", "<USER>", text)
    text = re.sub(r"/?r/[\w-]+", "<SUB>", text)
    text = text.lower()
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip()

# ✅ 1. Parquet 다운로드
save_dir = "tldr-17-parquet/train"
os.makedirs(save_dir, exist_ok=True)

import shutil  # ← 추가 필요

for i in range(10):
    idx = str(i).zfill(4)
    file_path = hf_hub_download(
        repo_id="webis/tldr-17",
        filename=f"default/partial-train/{idx}.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet"
    )
    local_path = os.path.join(save_dir, f"{idx}.parquet")
    shutil.copy(file_path, local_path)
    print(f"{idx}.parquet → {file_path}")

print("✅ Parquet 파일 다운로드 완료")

# ✅ 2. reddit.jsonl 생성
with open("reddit.jsonl", "w", encoding="utf-8") as fout:
    for fname in tqdm(os.listdir(save_dir)):
        if not fname.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(save_dir, fname))
        for _, row in df.iterrows():
            author = row.get("author")
            body = row.get("normalizedBody")
            if not author or not body:
                continue
            tokens = tokenizer.tokenize(preprocess(body))
            if len(tokens) < 5:
                continue
            fout.write(json.dumps({"author": author, "body": " ".join(tokens)}, ensure_ascii=False) + "\n")

print("✅ reddit.jsonl 생성 완료")

# dataset.py
import os
import sys
import json
import pickle
from collections import defaultdict, Counter

# Reddit 데이터셋 설정
INPUT_FILE = "reddit.jsonl"      # 원본 Reddit JSONL 파일
OUTPUT_DIR = "data"              # 전처리된 데이터 저장 디렉토리
VOCAB_SIZE = 10000               # 논문 기준 어휘 크기
MAX_WORDS_PER_CLIENT = 5000      # 클라이언트당 최대 단어 수
UNROLL_LENGTH = 10               # LSTM unroll 길이
TEST_POSTS = 100000              # 테스트 샘플 수

def validate_dataset():
    """
    데이터셋 빌드 전 raw 파일과 디렉토리 존재 여부 검증
    """
    if not os.path.isfile(INPUT_FILE):
        sys.exit(f"[Error] Reddit JSONL input file not found: {INPUT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_dataset():
    validate_dataset()

    # 1) 사용자별 단어 수집
    author_words = defaultdict(list)
    with open(INPUT_FILE, "r") as f:
        for line in f:
            post = json.loads(line)
            user = post.get("author") or post.get("user_id") or "<UNK>"
            text = post.get("body", "")
            words = text.split()
            author_words[user].extend(words)

    # 2) 클라이언트별 단어 제한
    train_clients = {
        user: words[:MAX_WORDS_PER_CLIENT]
        for user, words in author_words.items()
        if len(words) >= UNROLL_LENGTH
    }

    # 3) 어휘(vocabulary) 구성
    counter = Counter()
    for words in train_clients.values():
        counter.update(words)
    most_common = [w for w, _ in counter.most_common(VOCAB_SIZE - 1)]
    vocab = {w: idx + 1 for idx, w in enumerate(most_common)}
    vocab["<UNK>"] = 0

    # 4) 학습 데이터 시퀀스 생성
    train_data = {}
    for user, words in train_clients.items():
        seqs, tgts = [], []
        for i in range(len(words) - UNROLL_LENGTH):
            seq = words[i : i + UNROLL_LENGTH]
            tgt = words[i + UNROLL_LENGTH]
            seqs.append([vocab.get(w, 0) for w in seq])
            tgts.append(vocab.get(tgt, 0))
        train_data[user] = (seqs, tgts)

    # 5) 테스트 데이터 생성 (학습에 사용되지 않은 사용자 포스트에서 샘플링)
    test_seqs, test_tgts = [], []
    count = 0
    with open(INPUT_FILE, "r") as f:
        for line in f:
            if count >= TEST_POSTS:
                break
            post = json.loads(line)
            user = post.get("author") or post.get("user_id") or "<UNK>"
            if user in train_data:
                continue
            words = post.get("body", "").split()
            if len(words) <= UNROLL_LENGTH:
                continue
            seq = words[:UNROLL_LENGTH]
            tgt = words[UNROLL_LENGTH]
            test_seqs.append([vocab.get(w, 0) for w in seq])
            test_tgts.append(vocab.get(tgt, 0))
            count += 1

    # 6) 전처리 결과 저장
    with open(os.path.join(OUTPUT_DIR, "train_data.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(OUTPUT_DIR, "test_data.pkl"), "wb") as f:
        pickle.dump((test_seqs, test_tgts), f)
    with open(os.path.join(OUTPUT_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

def load_data(data_dir="data"):
    with open(os.path.join(data_dir, "train_data.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, "test_data.pkl"), "rb") as f:
        test_data = pickle.load(f)
    return {"train": train_data, "test": test_data}

if __name__ == "__main__":
    build_dataset()
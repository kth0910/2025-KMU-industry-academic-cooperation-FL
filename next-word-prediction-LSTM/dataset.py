# dataset.py (streaming + sharding, memory-safe)
import os
import sys
import json
import gzip
import pickle
import math
from collections import Counter, deque, defaultdict
from tqdm import tqdm

# ===== Config =====
INPUT_FILE = "reddit.jsonl"        # 원본 JSONL (fields: author|user_id, body)
OUTPUT_DIR = "data"                # 출력 디렉토리 루트
VOCAB_SIZE = 10000                 # <UNK> 포함 10k (실제 top-9999 + <UNK>)
MAX_WORDS_PER_CLIENT = 5000        # 클라이언트당 단어 상한
UNROLL_LENGTH = 10                 # LSTM unroll
TEST_POSTS = 100000                # 테스트 샘플 개수 목표
SHARD_MAX_LINES = 500_000          # 학습 샘플 샤드당 최대 라인 수
SHARD_DIR = "train_shards"         # 학습 샤드 폴더

UNK = "<UNK>"

def _ensure_dirs():
    if not os.path.isfile(INPUT_FILE):
        sys.exit(f"[Error] Reddit JSONL input file not found: {INPUT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, SHARD_DIR), exist_ok=True)
    
# dataset.py 맨 위쪽에 추가
def validate_dataset():
    """데이터셋 디렉토리와 원본 파일 존재 여부 검증."""
    _ensure_dirs()

def build_vocab():
    """
    Pass 1: 전체 코퍼스 1회 스캔 → 전역 단어 빈도 Counter → vocab.pkl 생성
    메모리: Counter만 유지
    """
    counter = Counter()
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="🔢 Pass1: Vocab counting"):
            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (post.get("body") or "").strip()
            if not text:
                continue
            words = text.split()
            counter.update(words)

    # top-(VOCAB_SIZE-1) + UNK=0
    most_common = [w for w, _ in counter.most_common(VOCAB_SIZE - 1)]
    vocab = {UNK: 0}
    for i, w in enumerate(most_common, start=1):
        vocab[w] = i

    with open(os.path.join(OUTPUT_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    return vocab

def _open_new_shard(shard_idx):
    path = os.path.join(OUTPUT_DIR, SHARD_DIR, f"train_{shard_idx:05d}.tsv.gz")
    fh = gzip.open(path, "wt", encoding="utf-8")
    return fh, path

def generate_train_shards(vocab):
    """
    Pass 2: 다시 스캔 → per-user deque(길이=UNROLL_LENGTH)와 사용 단어 수만 유지
            윈도우가 찰 때마다 즉시 라인으로 기록: user \t "id1 id2 ... id10" \t tgt
    출력: data/train_shards/*.tsv.gz, data/manifest.json
    """
    # per-user 상태(가벼움)
    buffers = defaultdict(lambda: deque(maxlen=UNROLL_LENGTH))  # 최근 UNROLL_LENGTH 토큰
    used_counts = defaultdict(int)                              # 유저별 이미 사용한 단어 수 (<= 5000)
    train_users = set()                                         # 학습에 1개 이상 배출한 유저

    shard_idx = 1
    shard_lines = 0
    total_lines = 0
    shard_fh, shard_path = _open_new_shard(shard_idx)
    shard_paths = [shard_path]

    def flush_if_needed():
        nonlocal shard_idx, shard_lines, shard_fh, shard_paths
        if shard_lines >= SHARD_MAX_LINES:
            shard_fh.close()
            shard_idx += 1
            shard_fh, path = _open_new_shard(shard_idx)
            shard_paths.append(path)
            shard_lines = 0

    unk_id = vocab.get(UNK, 0)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="🧠 Pass2: Train shards"):
            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = post.get("author") or post.get("user_id") or "<UNK_USER>"
            text = (post.get("body") or "").strip()
            if not text:
                continue
            if used_counts[user] >= MAX_WORDS_PER_CLIENT:
                continue

            # 이번 포스트에서 쓸 수 있는 남은 토큰 수
            remaining = MAX_WORDS_PER_CLIENT - used_counts[user]
            tokens = text.split()
            if not tokens:
                continue
            if len(tokens) > remaining:
                tokens = tokens[:remaining]

            # 토큰 → ids
            for w in tokens:
                wid = vocab.get(w, unk_id)
                buf = buffers[user]
                if len(buf) == UNROLL_LENGTH:
                    # 샘플 배출
                    seq_ids = " ".join(str(x) for x in buf)
                    shard_fh.write(f"{user}\t{seq_ids}\t{wid}\n")
                    shard_lines += 1
                    total_lines += 1
                    train_users.add(user)
                    flush_if_needed()
                buf.append(wid)
                used_counts[user] += 1

    shard_fh.close()

    # manifest 저장
    manifest = {
        "train_shards": shard_paths,
        "num_train_samples": total_lines,
        "num_train_users": len(train_users),
        "unroll_length": UNROLL_LENGTH,
        "vocab_size": len(vocab),
        "max_words_per_client": MAX_WORDS_PER_CLIENT,
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return train_users, total_lines, shard_paths

def build_test_set(vocab, train_users):
    """
    Pass 3: 학습에 한 번도 쓰이지 않은 사용자에서 TEST_POSTS 샘플 수집
            간단히 각 포스트의 앞 UNROLL_LENGTH+1 토큰만 사용
    출력: data/test.tsv.gz (user \t "id1..id10" \t tgt)
    """
    out_path = os.path.join(OUTPUT_DIR, "test.tsv.gz")
    written = 0
    unk_id = vocab.get(UNK, 0)

    with gzip.open(out_path, "wt", encoding="utf-8") as out_f, \
         open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="🧪 Pass3: Test set"):
            if written >= TEST_POSTS:
                break
            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = post.get("author") or post.get("user_id") or "<UNK_USER>"
            if user in train_users:
                continue
            words = (post.get("body") or "").split()
            if len(words) <= UNROLL_LENGTH:
                continue
            seq = words[:UNROLL_LENGTH]
            tgt = words[UNROLL_LENGTH]
            seq_ids = " ".join(str(vocab.get(w, unk_id)) for w in seq)
            tgt_id = vocab.get(tgt, unk_id)
            out_f.write(f"{user}\t{seq_ids}\t{tgt_id}\n")
            written += 1

    return out_path, written

def build_dataset():
    _ensure_dirs()
    vocab = build_vocab()
    train_users, n_samples, shard_paths = generate_train_shards(vocab)
    test_path, n_test = build_test_set(vocab, train_users)

    # 요약 로그
    print("\n✅ Done.")
    print(f"- Vocab: {len(vocab):,} (saved to {os.path.join(OUTPUT_DIR,'vocab.pkl')})")
    print(f"- Train samples: {n_samples:,} across {len(shard_paths)} shard(s)")
    print(f"- Test samples:  {n_test:,} (file: {test_path})")
    print(f"- Manifest:      {os.path.join(OUTPUT_DIR,'manifest.json')}")

# -------- Optional: simple loaders --------
def iter_train_samples(data_dir="data"):
    """
    샤드 파일들을 라인 단위로 스트리밍해서 (user, seq_ids[list[int]], tgt_id[int]) 생성
    """
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        mani = json.load(f)
    for p in mani["train_shards"]:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                user, seq_str, tgt_str = line.rstrip("\n").split("\t")
                seq = [int(x) for x in seq_str.split()]
                tgt = int(tgt_str)
                yield user, seq, tgt

def iter_test_samples(data_dir="data"):
    """
    test.tsv.gz를 라인 단위로 스트리밍해서 (user, seq_ids[list[int]], tgt_id[int]) 생성
    """
    path = os.path.join(data_dir, "test.tsv.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            user, seq_str, tgt_str = line.rstrip("\n").split("\t")
            seq = [int(x) for x in seq_str.split()]
            tgt = int(tgt_str)
            yield user, seq, tgt

if __name__ == "__main__":
    build_dataset()

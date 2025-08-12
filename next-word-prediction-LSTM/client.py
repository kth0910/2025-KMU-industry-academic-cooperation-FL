# client.py — streaming/sharded dataset용 클라이언트 (drop-in 교체본)

import os
import gzip
import flwr as fl
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from model import LSTMNextWordModel
from dataset import iter_train_samples, iter_test_samples  # streaming iterators


import os, torch
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


# ---------- Iterable datasets ----------
class UserIterableDataset(IterableDataset):
    """
    특정 client id(cid)에 해당하는 샘플만 스트리밍.
    최초 1회는 train_shards 전체를 훑어 내 cid만 캐시(tsv.gz)에 저장하고,
    이후 라운드부터는 캐시에서 바로 읽음.
    """
    def __init__(self, cid: str, data_dir: str = "data", cache: bool = True):
        self.cid = cid
        self.data_dir = data_dir
        self.cache = cache
        self.cache_path = os.path.join(data_dir, "cache", "train_user", f"{cid}.tsv.gz")

    def __iter__(self):
        # 캐시에서 읽기
        if self.cache and os.path.exists(self.cache_path):
            with gzip.open(self.cache_path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    _, seq_str, tgt_str = line.rstrip("\n").split("\t")
                    seq = [int(x) for x in seq_str.split()]
                    tgt = int(tgt_str)
                    yield seq, tgt
            return

        # 캐시가 없다면 샤드 전체를 훑어서 내 cid만 필터 → 캐시 생성
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        out = gzip.open(self.cache_path, "wt", encoding="utf-8") if self.cache else None
        try:
            for user, seq, tgt in iter_train_samples(self.data_dir):
                if user == self.cid:
                    if out:
                        out.write(f"{user}\t{' '.join(map(str, seq))}\t{tgt}\n")
                    yield seq, tgt
        finally:
            if out:
                out.close()


class TestIterableDataset(IterableDataset):
    """테스트 세트를 스트리밍. 필요시 limit로 상한 조절."""
    def __init__(self, data_dir: str = "data", limit: int | None = None):
        self.data_dir = data_dir
        self.limit = limit

    def __iter__(self):
        n = 0
        for _, seq, tgt in iter_test_samples(self.data_dir):
            yield seq, tgt
            n += 1
            if self.limit and n >= self.limit:
                break


def collate_fn(batch):
    import torch
    xs = [torch.tensor(seq, dtype=torch.long) for seq, tgt in batch]
    ys = [torch.tensor(tgt, dtype=torch.long) for seq, tgt in batch]
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


# ---------- Flower client ----------
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: str, device: torch.device | None = None):
        self.cid = cid
        self.device = torch.device("cuda")
        self.model = LSTMNextWordModel().to(self.device)

        # 스트리밍 데이터셋 (한 번 캐시되면 이후 라운드 빠름)
        self.train_ds = UserIterableDataset(cid, data_dir="data", cache=True)
        self.test_ds = TestIterableDataset(data_dir="data", limit=None)

    # --- Flower hooks ---
    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = float(config.get("lr", 0.1))
        epochs = int(config.get("epochs", 1))
        batch_size = int(config.get("batch_size", 256))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.model.train()

        train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=False,            # IterableDataset에서는 False 권장
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

        num_examples = 0
        for _ in range(epochs):
            for xb, yb in tqdm(train_loader, desc=f"[Client {self.cid}] Training", leave=True):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                num_examples += yb.size(0)

        return self.get_parameters(), num_examples, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        test_loader = DataLoader(
            self.test_ds,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, _ = self.model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = (correct / total) if total else 0.0
        return float(acc), total, {"accuracy": acc}

_TRAIN_USERS = None
def _load_users(path="data/train_users.txt"):
    global _TRAIN_USERS
    if _TRAIN_USERS is None:
        with open(path, "r", encoding="utf-8") as f:
            _TRAIN_USERS = [ln.strip() for ln in f if ln.strip()]
    return _TRAIN_USERS

def client_fn(cid: str) -> fl.client.Client:
    # Flower simulation에서 호출되는 팩토리

    users = _load_users()
    # 숫자 cid를 실제 user로 매핑 (모듈로 순환)
    try:
        idx = int(cid) % len(users)
    except:
        idx = 0
    real_user = users[idx]
    return FLClient(real_user).to_client()

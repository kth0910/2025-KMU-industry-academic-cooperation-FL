# make_train_users.py
import gzip, os, json

ROOT = "data/train_shards"
out = open(os.path.join("data", "train_users.txt"), "w", encoding="utf-8")
seen = set()
for fn in sorted(os.listdir(ROOT)):
    if not fn.endswith(".tsv.gz"):
        continue
    with gzip.open(os.path.join(ROOT, fn), "rt", encoding="utf-8") as fh:
        for line in fh:
            u = line.split("\t", 1)[0]
            if u not in seen:
                seen.add(u)
                out.write(u + "\n")
out.close()
print(f"users: {len(seen)} -> data/train_users.txt")

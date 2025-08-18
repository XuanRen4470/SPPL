import json

in_file = "/gpfs/users/a1796450/ACL_2024/SPPL/dataset/SQUAD/train.json"
out_file = "/gpfs/users/a1796450/ACL_2024/SPPL/dataset/SQUAD/train.json"

# 读取原始 JSON
with open(in_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 取前1000个
subset = data[:1000]

# 保存新文件
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print(f"Saved {len(subset)} samples to {out_file}")

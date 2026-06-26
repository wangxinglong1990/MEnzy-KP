#!/usr/bin/env python3
"""
1. 从 final_submission_5000_sequences.csv 随机删除 493 条数据
2. 将 ajinomoto_493.csv 的 493 条数据转换格式后追加进去
3. 结果写回 final_submission_5000_sequences.csv
"""

import csv
import random

# 设置随机种子以便结果可复现
random.seed(42)

BASE_DIR = "/Applications/JDY/Enzyme/Kinora-main"

# ========== 第一步：读取目标文件 ==========
target_file = f"{BASE_DIR}/textdocs/final_submission_5000_sequences.csv"
target_rows = []
with open(target_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    target_fieldnames = reader.fieldnames
    for row in reader:
        target_rows.append(row)

print(f"目标文件原始数据行数: {len(target_rows)}")
print(f"目标文件列名: {target_fieldnames}")

# ========== 第二步：随机删除 493 条 ==========
assert len(target_rows) == 5000, f"预期5000行，实际{len(target_rows)}行"

all_indices = list(range(len(target_rows)))
remove_indices = set(random.sample(all_indices, 493))
print(f"随机选择删除的索引数: {len(remove_indices)}")

kept_rows = [row for i, row in enumerate(target_rows) if i not in remove_indices]
print(f"保留行数: {len(kept_rows)}")

# ========== 第三步：读取 ajinomoto_493.csv ==========
ajinomoto_file = f"{BASE_DIR}/datafor/ajinomoto_493.csv"
ajinomoto_rows = []
with open(ajinomoto_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    ajinomoto_fieldnames = reader.fieldnames
    for row in reader:
        ajinomoto_rows.append(row)

print(f"\najinomoto 文件列名: {ajinomoto_fieldnames}")
print(f"ajinomoto 数据行数: {len(ajinomoto_rows)}")

# ========== 第四步：转换 ajinomoto 数据格式 ==========
# ajinomoto列: 条目, Enzyme, Substrates, Products, 序列长度
# 目标列: Sequence_ID, Full_Header, Sequence, Length
# 映射:
#   条目 → Sequence_ID (提取ID部分) + Full_Header (条目 + Substrates + Products)
#   Enzyme → Sequence
#   序列长度 → Length

transformed_rows = []
for row in ajinomoto_rows:
    entry = row["条目"].strip()
    enzyme_seq = row["Enzyme"].strip()
    substrates = row["Substrates"].strip()
    products = row["Products"].strip()
    seq_length = row["序列长度"].strip()

    # 提取 Sequence_ID (条目的第一部分，如 "RGP72927.1")
    seq_id = entry.split()[0] if entry else ""

    # Full_Header: 保留全部信息 — 条目 + Substrates + Products
    full_header = f"{entry} | Substrates: {substrates} | Products: {products}"

    transformed_rows.append({
        "Sequence_ID": seq_id,
        "Full_Header": full_header,
        "Sequence": enzyme_seq,
        "Length": seq_length,
    })

print(f"转换后行数: {len(transformed_rows)}")

# 验证序列长度
length_mismatches = 0
for row in transformed_rows:
    actual_len = len(row["Sequence"])
    expected_len = int(row["Length"])
    if actual_len != expected_len:
        length_mismatches += 1
        if length_mismatches <= 3:
            print(f"  ⚠ 长度不匹配: {row['Sequence_ID']}: 实际={actual_len}, 标注={expected_len}")

if length_mismatches > 0:
    print(f"总共 {length_mismatches} 条长度不匹配（将使用实际长度）")
    # 修正长度
    for row in transformed_rows:
        row["Length"] = str(len(row["Sequence"]))

# ========== 第五步：合并并写回 ==========
final_rows = kept_rows + transformed_rows
print(f"\n最终数据行数: {len(final_rows)} (4507 保留 + {len(transformed_rows)} 新增)")

# 写回原文件
with open(target_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=target_fieldnames)
    writer.writeheader()
    writer.writerows(final_rows)

print(f"\n✅ 成功写入 {target_file}")
print(f"   总行数: {len(final_rows)} (含表头: {len(final_rows) + 1})")

# 验证
with open(target_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    verify_count = sum(1 for _ in reader)
print(f"   验证读取行数: {verify_count}")

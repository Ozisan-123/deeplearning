import os
import xml.etree.ElementTree as ET

def extract_text(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    texts = []
    for seg in root.iter("seg"):
        if seg.text:
            texts.append(seg.text.strip())

    return texts


def process_folder(folder_path, out_zh="train.zh", out_en="train.en"):
    zh_lines = []
    en_lines = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".xml"):
            continue

        full_path = os.path.join(folder_path, filename)

        # 判断语言
        if ".zh.xml" in filename:
            print(f"Processing zh: {filename}")
            zh_lines.extend(extract_text(full_path))

        elif ".en.xml" in filename:
            print(f"Processing en: {filename}")
            en_lines.extend(extract_text(full_path))

    # 对齐检查（非常重要）
    min_len = min(len(zh_lines), len(en_lines))
    zh_lines = zh_lines[:min_len]
    en_lines = en_lines[:min_len]

    # 写文件
    with open(out_zh, "w", encoding="utf-8") as f:
        for line in zh_lines:
            f.write(line + "\n")

    with open(out_en, "w", encoding="utf-8") as f:
        for line in en_lines:
            f.write(line + "\n")

    print(f"Done. Total pairs: {min_len}")


# 使用
process_folder("dataset")
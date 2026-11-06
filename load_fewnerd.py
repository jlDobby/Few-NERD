import os
from collections import Counter

def load_fewnerd_file(file_path: str):
    """
    读取Few-NERD的txt文件（train/dev/test.txt）
    返回：sentences: List[List[tokens]], tags: List[List[labels]]
    """
    sentences = []
    tag_seqs = []
    cur_tokens = []
    cur_tags = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 空行 = 句子结束
            if not line:
                if cur_tokens:
                    sentences.append(cur_tokens)
                    tag_seqs.append(cur_tags)
                    cur_tokens = []
                    cur_tags = []
                continue
            # 按制表符分割 token 和标签
            token, tag = line.split("\t")
            cur_tokens.append(token)
            cur_tags.append(tag)
        # 处理文件最后一句（防止末尾没有空行）
        if cur_tokens:
            sentences.append(cur_tokens)
            tag_seqs.append(cur_tags)
    return sentences, tag_seqs


def load_all_fewnerd(base_dir="data"):
    """一次性加载 inter / intra / supervised 三个划分"""
    splits = {
        "inter": os.path.join(base_dir, "inter"),
        "intra": os.path.join(base_dir, "intra"),
        "supervised": os.path.join(base_dir, "supervised"),
    }
    dataset = {}
    for split_name, folder in splits.items():
        dataset[split_name] = {}
        for part in ["train", "dev", "test"]:
            fp = os.path.join(folder, f"{part}.txt")
            tokens, tags = load_fewnerd_file(fp)
            dataset[split_name][part] = {"tokens": tokens, "tags": tags}
    return dataset


if __name__ == "__main__":
    data = load_all_fewnerd(base_dir="data")

    # 打印统计信息
    for setting in ["inter", "intra", "supervised"]:
        print(f"\n===== {setting} setting =====")
        all_tags = []
        for split in ["train", "dev", "test"]:
            sent_list = data[setting][split]["tokens"]
            tag_list = data[setting][split]["tags"]
            print(f"  {split}: 句子数量 = {len(sent_list)}")
            for seq in tag_list:
                all_tags.extend(seq)
        tag_counter = Counter(all_tags)
        print(f"  标签统计（前15）: {tag_counter.most_common(15)}")

    # 打印一条样例（supervised的train第一条）
    print("\n===== 样例句子 supervised/train 第0条 =====")
    sample_tokens = data["supervised"]["train"]["tokens"][0]
    sample_tags = data["supervised"]["train"]["tags"][0]
    for w, t in zip(sample_tokens, sample_tags):
        print(f"{w:15s} {t}")

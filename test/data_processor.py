import pandas as pd
from pathlib import Path
import json
from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# --- 配置项 (Configuration) ---
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets"
GEN_DIR = DATA_DIR / "gen"
REAL_DIR = DATA_DIR / "real"
# 输出文件位于 test 目录下
OUTPUT_FILE = ROOT / "test" / "prompts_tokens.json"
SD_MODEL = "CompVis/stable-diffusion-v1-4" # 用于获取tokenizer的模型

# --- Tokenizer 相关函数 (Tokenizer Functions) ---

def setup_tokenizer(model_id: str):
    """加载并返回指定模型的tokenizer"""
    print(f"正在从 {model_id} 加载分词器...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    return pipe.tokenizer

def find_token_indices(prompt: str, words_to_find: list, tokenizer) -> list:
    """在prompt中查找指定词汇的token索引"""
    indices = []
    prompt_token_ids = tokenizer.encode(prompt)
    for word in words_to_find:
        word_token_ids = tokenizer.encode(word, add_special_tokens=False)
        for i in range(len(prompt_token_ids) - len(word_token_ids) + 1):
            if prompt_token_ids[i:i + len(word_token_ids)] == word_token_ids:
                indices.append(list(range(i, i + len(word_token_ids))))
                break
    return indices

# --- 核心处理函数模板 (Core Processing Templates) ---

def process_dataset(
    csv_path: Path,
    prompt_column: str,
    words_to_find: List[str],
    tokenizer: Any,
    data_dict: Dict,
    key_creation_logic: callable,
    n_samples: Optional[int] = None,
    description: str = ""
):
    """
    一个通用的数据集处理函数.

    参数:
    - csv_path: 数据集CSV文件的路径.
    - prompt_column: 包含提示词的列名.
    - words_to_find: 需要查找和擦除的关键词列表.
    - tokenizer: 分词器实例.
    - data_dict: 用于存储最终结果的字典 (会直接被修改).
    - key_creation_logic: 一个函数，用于根据行数据生成字典的键.
    - n_samples: 需要抽样的数量，如果为None则处理全部数据.
    - description: tqdm进度条的描述文字.
    """
    if not csv_path.exists():
        print(f"警告: 文件 {csv_path} 不存在，跳过处理。")
        return

    print(f"正在处理 {csv_path}...")
    df = pd.read_csv(csv_path)

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=2025)

    for index, row in tqdm(df.iterrows(), total=len(df), desc=description):
        prompt = row[prompt_column]
        token_indices = find_token_indices(prompt, words_to_find, tokenizer)

        if token_indices:
            # 使用传入的逻辑来创建key
            key = key_creation_logic(row, index)
            data_dict[key] = {
                "prompt": prompt,
                "token_indices": token_indices,
                "mode": "neg"
            }

def gen_prep(
    csv_path: Path,
    prompt_column: str,
    words_to_find: List[str],
    tokenizer: Any,
    data_dict: Dict,
    n_samples: Optional[int] = None,
    description: str = "",
    key_prefix: str = "gen",
    id_column: Optional[str] = None
):
    """
    【模板】用于处理“生成图像”的数据集.
    它会自动构建一个唯一的键 (key), 例如: 'gen_vangogh_0.jpg'.
    """
    def create_key(row, index):
        # 如果指定了id_column，则用它，否则用行索引
        unique_id = row[id_column] if id_column and id_column in row else index
        return f"{key_prefix}_{unique_id}.jpg"

    process_dataset(csv_path, prompt_column, words_to_find, tokenizer, data_dict, create_key, n_samples, description)

def real_prep(
    csv_path: Path,
    prompt_column: str,
    key_column: str,
    words_to_find: List[str],
    tokenizer: Any,
    data_dict: Dict,
    n_samples: Optional[int] = None,
    description: str = ""
):
    """
    【模板】用于处理“真实图像”的数据集.
    它会从指定的列 (key_column) 中提取文件名作为键 (key).
    """
    def create_key(row, index):
        # 从指定列获取路径，并提取文件名
        return Path(row[key_column]).name

    process_dataset(csv_path, prompt_column, words_to_find, tokenizer, data_dict, create_key, n_samples, description)


# --- 主函数 (Main Function) ---

def main():
    """主函数，使用模板来生成prompts_tokens.json"""
    tokenizer = setup_tokenizer(SD_MODEL)
    final_data = {}
    # --- 新增步骤：在开始前加载已有数据 ---
    if OUTPUT_FILE.exists():
        print(f"发现已存在的 {OUTPUT_FILE}，将加载内容进行更新...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                final_data = json.load(f)
            except json.JSONDecodeError:
                print(f"警告: {OUTPUT_FILE} 文件内容不是有效的JSON格式，将创建一个新文件。")
                final_data = {} # 如果文件为空或损坏，则重置为空字典
    # ------------------------------------
    print("\n--- 任务1: 处理真实图像数据集 ---")
    
    # 1a. 真实图像 - COCO验证集中的汽车
    real_prep(
        csv_path=REAL_DIR / "coco/cocoval17_car_dataset.csv",
        prompt_column="prompt",
        key_column="path",  # 文件名从此列获取
        words_to_find = ["car"],
        tokenizer=tokenizer,
        data_dict=final_data,
        n_samples=20,
        description="COCO(Cars) Real Processing"
    )
    print("\n--- 任务2: 处理生成图像数据集 ---")
    # 2a. 生成图像 - 梵高风格
    gen_prep(
        csv_path=GEN_DIR / "vangogh_prompts.csv",
        prompt_column="prompt",
        words_to_find=["Vincent van Gogh"],
        tokenizer=tokenizer,
        data_dict=final_data,
        n_samples=20,
        description="Van Gogh Processing",
        key_prefix="gen_vangogh"
    )

    # 2b. 生成图像 - Tyler Edlin风格
    gen_prep(
        csv_path=GEN_DIR / "tyler_prompts.csv",
        prompt_column="prompt",
        words_to_find=["Tyler Edlin"],
        tokenizer=tokenizer,
        data_dict=final_data,
        n_samples=10,
        description="Tyler Edlin Processing",
        key_prefix="gen_tyler"
    )

    # 2c. 生成图像 - COCO数据集中的汽车
    gen_prep(
        csv_path=REAL_DIR / "coco/cocoval17_car_dataset.csv",
        prompt_column="prompt",
        words_to_find=["car"],
        tokenizer=tokenizer,
        data_dict=final_data,
        n_samples=20,
        description="COCO(Cars) Gen Processing",
        key_prefix="gen_coco_car",
        id_column="image_id"
    )


    # 写入最终的JSON文件
    print(f"\n正在将 {len(final_data)} 条数据写入 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print("处理完成！")

if __name__ == "__main__":
    main()
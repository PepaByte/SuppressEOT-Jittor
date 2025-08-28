import json
import os
import re

def create_candidates_for_result_files_only():
    """
    Specifically finds files ending with '-result.jpg', matches them with prompts,
    and generates the correct candidate files with extension-less keys
    required by clipscore.py.
    """
    # --- 配置区域 (路径已更新) ---
    PROMPT_FILE = 'prompts_tokens.json'
    IMAGE_ROOT_DIRS = ['results/gen', 'results/real'] # <--- 已修改
    OUTPUT_DIR = 'grouped_candidates_final_results'
    # --- 结束配置 ---

    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 提示词文件 '{PROMPT_FILE}' 未找到。")
        return

    # 创建一个从基础文件名到提示词的查找表
    base_name_to_prompt = {
        os.path.splitext(key)[0]: value['prompt']
        for key, value in prompt_data.items()
    }

    grouped_candidates = {}

    print("正在扫描目录，仅查找 '-result.jpg' 文件...")
    for directory in IMAGE_ROOT_DIRS:
        if not os.path.isdir(directory):
            print(f"警告: 目录 '{directory}' 不存在，已跳过。")
            continue

        for filename in os.listdir(directory):
            if not filename.endswith('-result.jpg'):
                continue

            matched_base = None
            for base_name in base_name_to_prompt.keys():
                if filename.startswith(base_name):
                    matched_base = base_name
                    break
            
            if matched_base:
                prompt = base_name_to_prompt[matched_base]
                
                group_name = 'unknown'
                if matched_base.startswith('gen_coco_car'):
                    group_name = 'gen_coco_car'
                elif matched_base.startswith('gen_tyler'):
                    group_name = 'gen_tyler'
                elif matched_base.startswith('gen_vangogh'):
                    group_name = 'gen_vangogh'
                elif matched_base.startswith('unsplash'):
                    group_name = 'unsplash'
                elif re.match(r'^\d+$', matched_base):
                    group_name = 'real_coco_car'

                if group_name not in grouped_candidates:
                    grouped_candidates[group_name] = {}
                
                key_without_extension = os.path.splitext(filename)[0]
                grouped_candidates[group_name][key_without_extension] = prompt
            else:
                print(f"警告: 无法为 '{filename}' 找到匹配的提示词，已忽略。")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not grouped_candidates:
        print("错误：在指定目录中未能找到任何 '-result.jpg' 文件。")
        return

    print(f"\n脚本已在 '{OUTPUT_DIR}/' 目录下为 '-result.jpg' 文件生成了以下 JSON:")
    for group_name, items in grouped_candidates.items():
        file_path = os.path.join(OUTPUT_DIR, f'candidates_{group_name}_results.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"- {file_path}")

    print("\n请使用以下命令为每个分组计算 CLIP 得分:")
    # --- 路径映射已更新 ---
    image_dirs_map = {
        'gen_coco_car': 'results/gen/',     # <--- 已修改
        'gen_tyler': 'results/gen/',      # <--- 已修改
        'gen_vangogh': 'results/gen/',    # <--- 已修改
        'unsplash': 'results/real/',      # <--- 已修改
        'real_coco_car': 'results/real/'  # <--- 已修改
    }
    for group_name in grouped_candidates.keys():
        json_path = os.path.join(OUTPUT_DIR, f'candidates_{group_name}_results.json')
        image_dir = image_dirs_map.get(group_name, '未知目录')
        print(f"\n# --- {group_name} (results) ---")
        print(f"python clipscore-main/clipscore.py \"{json_path}\" \"{image_dir}\"")


if __name__ == '__main__':
    create_candidates_for_result_files_only()
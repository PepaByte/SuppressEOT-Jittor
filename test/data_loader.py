import json
import subprocess
import sys
from pathlib import Path

# --- 路径设置 ---
# __file__ 指向当前文件 (data_loader.py)
# .resolve().parent.parent 会获取到 SuppressEOT-master-torch 这个根目录
ROOT = Path(__file__).resolve().parent.parent

# 定义各个图片数据源的目录
UNSPLASH_DIR = ROOT / "datasets" / "real" / "unsplash"
COCO_DIR = ROOT / "datasets" / "real" / "coco" / "val2017"

# --- 加载核心配置文件 ---
try:
    with open(ROOT / "test" / "prompts_tokens.json", 'r', encoding='utf-8') as f:
        META = json.load(f)
except FileNotFoundError:
    print(f"错误: 无法找到 prompts_tokens.json 文件，请先运行数据准备脚本。")
    sys.exit(1) # 退出程序
except json.JSONDecodeError:
    print(f"错误: prompts_tokens.json 文件格式不正确，请检查。")
    sys.exit(1)


def run(cmd):
    """健壮的 run 函数，用于调用外部脚本并捕获可能的错误"""
    try:
        # 为了日志清晰，打印将要执行的命令
        print(">>> " + " ".join(map(str, cmd)))
        # check=True 会在命令失败时抛出异常，并捕获输出
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        # 如果需要，可以取消注释下面两行来查看成功执行的输出
        # print(f"Output:\n{result.stdout}")
        # print(f"Error Output (if any):\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"--- 命令执行失败 ---")
        print(f"命令: {' '.join(e.cmd)}")
        print(f"返回码: {e.returncode}")
        print(f"标准输出:\n{e.stdout}")
        print(f"错误输出:\n{e.stderr}")
        print("-" * 20)
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 主处理逻辑 ---

# 遍历JSON文件中的每一个条目 (key, info)
for key, info in META.items():
    
    image_path = None
    task_type = None

    # 1. 根据 key 的前缀判断任务类型和图片路径
    if key.startswith("unsplash_"):
        task_type = "Real-Image"
        image_path = UNSPLASH_DIR / key
        print(f"\n--- 处理 Unsplash 真实图片: {key} ---")
        
    elif key.startswith("gen_"):
        task_type = "Generated-Image"
        print(f"\n--- 处理生成图片任务: {key} ---")

    else: # 默认情况，处理来自COCO的真实图片
        task_type = "Real-Image"
        image_path = COCO_DIR / key
        print(f"\n--- 处理 COCO 真实图片: {key} ---")

    # 检查真实图片文件是否存在
    if task_type == "Real-Image" and not image_path.exists():
        print(f"警告: 图片文件不存在，跳过。路径: {image_path}")
        continue
        
    # 2. 准备通用参数
    prompt = info["prompt"]
    # 将Python列表转换为JSON格式的字符串，以便作为命令行参数传递
    token_indices = json.dumps(info["token_indices"])
    mode = info.get("mode", "neg") # 默认为 "neg" 模式
    
    # 构造主脚本的路径
    script_path = str(ROOT / "suppress_eot_w_nulltext.py")

    # 3. 构建并执行命令
    if task_type == "Real-Image":
        # 构建处理真实图片的命令
        cmd = [
            sys.executable, script_path,
            "--type", "Real-Image",
            "--image_path", str(image_path),
            "--prompt", prompt,
            "--token_indices", token_indices,
            "--base_name", key
        ]
        if mode == 'pos':
            # 'pos' 模式下需要添加额外的 alpha 参数
            cmd.extend(["--alpha", "[-0.001,]"])
        
        run(cmd)

    elif task_type == "Generated-Image":
        # 构建处理生成图片的命令
        cmd = [
            sys.executable, script_path,
            "--type", "Generated-Image",
            "--prompt", prompt,
            "--token_indices", token_indices,
            "--seed", str(info.get("seed", 2025)),  # 如果json没提供seed，就用默认值
            "--iter_each_step", str(info.get("iter_each_step", 0)),
            "--base_name", key
        ]
        if mode == 'pos':
            cmd.extend(["--alpha", "[-0.001,]"])

        run(cmd)
    
    print("-" * 20)

print("\n所有任务处理完毕。")
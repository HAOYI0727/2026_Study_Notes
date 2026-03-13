"""
Name: Qwen3_PPO_Inference.py
Description: PPO训练后模型的独立推理脚本，支持交互式对话和批量测试两种模式，不包含记忆模块，每次对话独立处理
"""

# ====================== 1. 镜像加速与导入必要库 ======================

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)

# ====================== 2. 加载模型和分词器 ======================

def load_trained_model(model_path):
    """
    加载训练好的PPO模型和分词器
    参数：model_path: 训练好的模型保存路径
    返回：tokenizer: 分词器
         model: 加载的模型
    加载策略：
        1. 使用4-bit量化加载，大幅降低显存
        2. 自动设备映射，支持GPU/CPU
        3. 保持与训练时一致的配置
    """
    
    # 使用4-bit量化配置（与训练时保持一致）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 启用4-bit量化加载
        bnb_4bit_use_double_quant=True,       # 使用双重量化，进一步压缩
        bnb_4bit_quant_type="nf4",            # NF4量化类型，专门为神经网络优化
        bnb_4bit_compute_dtype=torch.bfloat16 # 计算时使用bfloat16精度
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",  # 因果语言模型需要右侧padding
        trust_remote_code=True
    )
    
    # 设置pad_token（Qwen模型通常没有默认pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载训练好的策略模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    # 设置为评估模式
    model.eval()
    
    print(f"✓ 模型加载成功: {model_path}")
    print(f"  - 设备: {model.device}")
    print(f"  - 参数量: ~4B (4-bit量化)")
    
    return tokenizer, model

# ====================== 3. 使用模型生成回复 ======================

def generate_response(tokenizer, model, question, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """
    使用训练好的模型生成回复
    参数：tokenizer: 分词器
         model: 训练好的模型
         question: 用户输入的问题
         max_new_tokens: 最大生成token数
         temperature: 温度参数，控制随机性
         top_p: 核采样参数，控制多样性
    返回：response: 模型生成的回答
    生成流程：
        1. 构建Qwen对话格式的prompt
        2. 编码输入文本
        3. 使用模型生成回答
        4. 解码并提取assistant部分
    """
    
    # 设置生成参数（显式设置Qwen模型的默认值以避免警告）
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,         # 最大生成长度
        temperature=temperature,               # 温度参数（>1更随机，<1更确定）
        top_p=top_p,                           # 核采样参数
        top_k=20,                              # Top-k采样（Qwen模型的默认值）
        do_sample=True,                        # 启用采样生成
        pad_token_id=tokenizer.eos_token_id,   # 填充标记
        eos_token_id=tokenizer.eos_token_id,   # 结束标记
        bos_token_id=151643                    # 开始标记（Qwen模型默认值）
    )
    
    # 构建Qwen对话格式
    # Qwen使用ChatML格式：<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回复（禁用梯度计算以节省显存）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # 解码输出（保留特殊标记以便提取assistant部分）
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
    
    # 提取assistant的回复部分
    if "<|im_start|>assistant" in generated_text:
        # 分割获取assistant标记后的内容
        response = generated_text.split("<|im_start|>assistant")[-1]
        # 移除结束标记
        response = response.replace("<|im_end|>", "").strip()
    else:
        # 如果格式异常，回退到简单处理
        response = generated_text.replace(prompt, "").strip()
    
    return response

# ====================== 4. 推理模式 ======================

def interactive_inference(model_path):
    """
    交互式推理模式
    特点：支持连续对话（但不保存历史）、实时生成回复、支持退出命令、错误处理机制
    """
    print("正在加载训练好的模型...")
    tokenizer, model = load_trained_model(model_path)
    print("模型加载完成！\n")
    
    print("=" * 60)
    print("PPO训练模型推理模式")
    print("=" * 60)
    print("使用说明：")
    print("  - 直接输入问题即可获得回答")
    print("  - 输入 'quit' 或 '退出' 结束对话")
    print("  - 支持中文问题")
    print("=" * 60)
    
    # 对话计数器
    conversation_count = 0
    
    while True:
        try:
            # 获取用户输入
            question = input("\n请输入您的问题: ").strip()
            # 检查退出命令
            if question.lower() in ['quit', '退出', 'exit', 'q']:
                print("\n感谢使用，再见！")
                break
            # 检查空输入
            if not question:
                print("问题不能为空，请重新输入。")
                continue
            
            conversation_count += 1
            print(f"\n[对话 {conversation_count}]")
            print(f"用户: {question}")
            print("正在生成回复...", end="", flush=True)
            
            # 生成回复
            response = generate_response(tokenizer, model, question)
            print("\r", end="")  # 清除"正在生成"提示
            print(f"助手: {response}")
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n生成过程中出现错误: {e}")
            print("请尝试重新输入问题。")
            continue

def batch_inference(model_path, questions):
    """
    批量推理模式
    参数：model_path: 模型路径
         questions: 问题列表
    特点：一次性处理多个问题、适合测试和评估、清晰的输出格式
    """
    print("正在加载训练好的模型...")
    tokenizer, model = load_trained_model(model_path)
    print("模型加载完成！\n")
    
    print("=" * 60)
    print("批量推理测试")
    print("=" * 60)
    print(f"测试问题数量: {len(questions)}")
    print("=" * 60)
    
    # 记录每个问题的生成时间（可选）
    import time
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}/{len(questions)}: {question}")
        print("-" * 50)
        
        # 计时开始
        start_time = time.time()
        # 生成回复
        response = generate_response(tokenizer, model, question)
        # 计时结束
        elapsed_time = time.time() - start_time
        
        print(f"回答: {response}")
        print(f"生成时间: {elapsed_time:.2f} 秒")
        print("-" * 50)
    
    print("\n批量推理完成！")
    print(f"总共处理 {len(questions)} 个问题")

# ====================== 5. 主函数 ======================

def main():
    """
    主函数：解析命令行参数并执行相应的推理模式
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PPO训练模型推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 批量推理模式（默认）
  python ppo_model_inference.py --model_path ./Qwen3_PPO_Reward_Model
  
  # 交互式对话模式
  python ppo_model_inference.py --model_path ./Qwen3_PPO_Reward_Model --mode interactive
  
  # 自定义批量测试问题
  python ppo_model_inference.py --model_path ./Qwen3_PPO_Reward_Model --mode batch --questions "你好" "什么是AI"
        """
    )
    
    parser.add_argument("--model_path", type=str, default="./Qwen3_PPO_Reward_Model", 
                       help="训练好的模型路径 (默认: ./Qwen3_PPO_Reward_Model)")
    
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], default="batch",
                       help="推理模式: interactive(交互式) 或 batch(批量) (默认: batch)")
    
    parser.add_argument("--questions", type=str, nargs="+", 
                       default=[
                           "什么是LoRA？",
                           "LoRA有什么优势？", 
                           "什么是PPO算法？",
                           "请用优美的话语夸赞和鼓励我"
                       ],
                       help="批量推理的问题列表 (默认: 预设的4个测试问题)")
    
    parser.add_argument("--max_tokens", type=int, default=256,
                       help="最大生成token数 (默认: 256)")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数，控制随机性 (默认: 0.7)")
    
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="核采样参数 (默认: 0.9)")
    
    args = parser.parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径 '{args.model_path}' 不存在！")
        print("请先运行训练脚本或指定正确的模型路径。")
        print("可能的解决方案：")
        print("  1. 检查路径是否正确")
        print("  2. 运行PPO训练脚本生成模型")
        print("  3. 使用默认路径 ./Qwen3_PPO_Reward_Model")
        exit(1)
    
    # 根据模式执行相应功能
    if args.mode == "interactive":
        print(f"启动交互式模式，模型路径: {args.model_path}")
        interactive_inference(args.model_path)
    else:
        print(f"启动批量推理模式，模型路径: {args.model_path}")
        print(f"将测试 {len(args.questions)} 个问题")
        batch_inference(args.model_path, args.questions)

if __name__ == "__main__":
    main()
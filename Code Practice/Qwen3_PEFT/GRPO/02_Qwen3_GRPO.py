"""
Name: Qwen3_GRPO.py
功能: 基于Qwen3-4B-Instruct模型的GRPO(Group Relative Policy Optimization)强化学习训练实现,
      包含4-bit量化、LoRA参数高效微调、多奖励函数组合、训练后推理测试等完整流程
"""

"""
GRPO (Group Relative Policy Optimization) 原理：
    1. 通过组内相对评估替代传统Critic模型，大幅降低显存占用
    2. 在每组生成中计算相对优势，更稳定地优化策略
    3. 结合KL散度约束，防止策略更新过快导致崩溃
    4. 特别适合大语言模型的RLHF训练
GRPO与PPO的核心区别：
    - PPO: 需要同时加载策略模型(Policy)、价值模型(Critic)和奖励模型(Reward)，显存占用大
    - GRPO: 只需要策略模型和奖励模型，通过组内生成样本的相对表现计算优势，显存节省约50%
""" 

# ====================== 1. 镜像加速与导入必要库 ======================

import os
import shutil
import torch
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息，保持输出整洁

# 设置镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 核心库
from datasets import Dataset  # 数据集处理
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    GenerationConfig
)
# TRL库相关导入 - GRPO是TRL 0.26.0版本新增的重要功能
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# ====================== 2. 自定义奖励函数 ======================

def length_reward_func(prompts, completions, **kwargs):
    """
    基于生成长度的奖励函数——奖励更长的回答（基于token数量），鼓励模型生成更详细的回答
    参数：
        - prompts: 输入的提示列表
        - completions: 模型生成的回答列表
        - **kwargs: 其他参数
    返回：
        - rewards: 长度奖励列表，值为浮点数
    """
    return [float(len(completion)) for completion in completions]

def content_quality_reward_func(prompts, completions, **kwargs):
    """
    基于内容质量的奖励函数——通过关键词匹配评估回答质量，奖励包含特定关键词的回答
    参数：
        - prompts: 输入的提示列表
        - completions: 模型生成的回答列表
        - **kwargs: 其他参数
    返回：
        - rewards: 内容质量奖励列表，值范围[0,1]
    """
    rewards = []
    # 定义与LoRA和微调相关的关键词
    quality_keywords = ['LoRA', '微调', '参数', '高效', '训练', '模型', '优势', '步骤']
    
    for completion in completions:
        # 计算关键词命中数量
        keyword_count = sum(1 for keyword in quality_keywords if keyword in completion)
        # 归一化到0-1范围
        reward = min(keyword_count / len(quality_keywords), 1.0)
        rewards.append(reward)
    
    return rewards

def format_reward_func(prompts, completions, **kwargs):
    """
    基于格式规范的奖励函数——检查回答是否符合Qwen对话格式，确保生成的回答格式正确
    参数：
        - prompts: 输入的提示列表
        - completions: 模型生成的回答列表
        - **kwargs: 其他参数
    返回：
        - rewards: 格式奖励列表，值范围[0,1]
    """
    rewards = []
    
    for completion in completions:
        # 检查是否包含正确的结束标记
        if '<|im_end|>' in completion and '<|im_start|>' in completion:
            # 检查格式完整性：开始和结束标记数量是否匹配
            if completion.count('<|im_start|>') == completion.count('<|im_end|>'):
                reward = 1.0
            else:
                reward = 0.5
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

# ====================== 3. 主函数 ======================

if __name__ == "__main__":
    # ====================== 3.1 GRPO训练参数配置 ======================
    """
    GRPOConfig参数，继承自TrainingArguments，增加了GRPO特有的训练参数
    核心参数：
        - num_generations: 每个提示生成的回答数量，用于计算组内相对优势
        - beta: KL散度系数，控制与参考模型的偏离程度（参考DeepSeek-R1设置）
        - epsilon: 裁剪参数，限制策略更新幅度，防止崩溃
        - loss_type: 损失函数类型，支持"dapo"、"grpo"等
        - scale_rewards: 奖励缩放策略，"group"表示在组内进行归一化
    """
    training_args = GRPOConfig(
        # 基础训练参数
        learning_rate=3e-6,                   # 学习率（RL训练通常比SFT小10倍）
        output_dir="./Qwen3_GRPO_checkpoint",   # 输出目录
        per_device_train_batch_size=1,        # 批次大小（GRPO通常使用小批次）
        gradient_accumulation_steps=4,        # 梯度累积步数
        num_train_epochs=3,                   # 训练轮数
        
        # GRPO特定参数
        num_generations=4,             # 每个提示生成4个回答用于组内比较
        max_completion_length=256,     # 最大生成长度
        temperature=0.7,               # 采样温度（>1更随机，<1更确定）
        top_p=0.9,                     # 核采样参数（保留概率和达0.9的token）
        
        # KL散度控制 - 防止策略偏离太远
        beta=0.001,                    # KL系数（参考DeepSeek-R1设置）
        epsilon=0.2,                   # 裁剪参数（类似PPO的clip范围）
        
        # 损失函数配置
        loss_type="dapo",              # 使用DAPO损失函数（DeepSeek优化的GRPO变体）
        scale_rewards="group",         # 在组内进行奖励归一化
        
        # 训练优化
        bf16=True,                     # 使用bfloat16混合精度
        gradient_checkpointing=True,   # 启用梯度检查点，用计算换显存
        
        # 日志和保存
        logging_steps=10,              # 每10步打印一次日志
        save_steps=100,                # 每100步保存一次检查点
        report_to="none"               # 不向外部服务报告
    )
    
    # ====================== 3.2 清理输出目录 ======================
    # 确保从干净的目录开始训练
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    print(f"已清理输出目录: {training_args.output_dir}")
    
    # ====================== 3.3 模型配置 ======================
    model_name = "Qwen/Qwen3-4B-Instruct-2507"  # 基础模型
    
    # 4-bit量化配置 - 大幅降低显存占用
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 启用4-bit量化
        bnb_4bit_use_double_quant=True,       # 双重量化（进一步压缩）
        bnb_4bit_quant_type="nf4",            # NF4量化类型（专门为神经网络优化）
        bnb_4bit_compute_dtype=torch.bfloat16 # 计算时使用bfloat16
    )
    
    # ====================== 3.4 分词器加载 ======================
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"    # 因果语言模型需要右侧padding
    )
    
    # 确保分词器有pad_token（Qwen模型通常没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("分词器加载完成")
    
    # ====================== 3.5 策略模型加载 ======================
    """
    策略模型(Policy Model)：要优化的生成模型
    负责根据prompt生成回答，并通过GRPO优化其生成策略
    """
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,  # 应用量化
        device_map="auto",                        # 自动设备分配
        trust_remote_code=True,                   # 信任远程代码
        dtype=torch.bfloat16                      # bfloat16精度
    )
    print("策略模型加载完成")
    
    # ====================== 3.6 奖励模型加载 ======================
    """
    奖励模型(Reward Model)：评估生成质量的模型,可以是预训练好的奖励模型，也可以是自定义的奖励函数
    这里优先尝试加载已训练好的奖励模型
    """
    reward_model_path = "./Qwen3_GRPO_Reward_Model"
    
    if os.path.exists(reward_model_path):
        print(f"加载已训练好的奖励模型: {reward_model_path}")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            num_labels=1,                            # 回归任务，输出单个分值
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
    else:
        print("警告：未找到训练好的奖励模型，将使用基础模型")
        # 将基础模型转换为序列分类模型
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,                            # 添加分类头
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
    
    print("奖励模型加载完成")
    
    # ====================== 3.7 LoRA配置 ======================
    """
    LoRA配置：参数高效微调，只训练一小部分参数，大幅减少计算量和显存需求
    """
    peft_config = LoraConfig(
        r=8,                                         # LoRA秩（越小参数越少）
        lora_alpha=16,                               # 缩放因子（通常为r的2倍）
        lora_dropout=0.05,                           # Dropout比例
        # 为了加快训练速度，这里只指定query层进行微调
        target_modules=["q_proj"],                   # 目标模块（实际应用可扩展到更多层）
        bias="none",                                 # 不训练偏置
        task_type="CAUSAL_LM"                        # 任务类型：因果语言模型
    )
    
    # ====================== 3.8 数据集准备 ======================
    """
    GRPO数据集格式要求：
    每个样本必须包含"prompt"字段，表示输入的问题或指令，无需提供答案，模型会通过强化学习自主探索最优回答
    """
    train_data = [
        {"prompt": "什么是LoRA？"},
        {"prompt": "LoRA有什么优势？"}, 
        {"prompt": "如何使用LoRA进行模型微调？"},
        {"prompt": "什么是PPO算法？"},
        {"prompt": "RLHF的全称是什么？"},
        {"prompt": "参数高效微调有哪些方法？"},
        {"prompt": "LoRA和全参数微调有什么区别？"},
        {"prompt": "如何选择合适的LoRA秩？"}
    ]
    train_dataset = Dataset.from_list(train_data)
    print(f"创建了 {len(train_data)} 个训练样本")
    
    # ====================== 3.9 创建GRPO训练器 ======================
    """
    GRPOTrainer创建流程：
        1. 将多个奖励函数组合成一个列表
        2. 奖励函数可以是预训练模型，也可以是自定义函数
        3. 训练器会自动计算每个生成的奖励值，并进行组内归一化
    奖励函数组合的优势：
        - 长度奖励：鼓励详细回答
        - 内容质量奖励：确保回答包含关键信息
        - 格式奖励：保证输出格式正确
        - 奖励模型：提供更精准的质量评估
    """
    # 定义奖励函数组合（按权重顺序添加）
    reward_functions = [
        reward_model,                    # 预训练奖励模型（权重最高）
        length_reward_func,              # 长度奖励（鼓励详细回答）
        content_quality_reward_func,     # 内容质量奖励（关键词匹配）
        format_reward_func               # 格式规范奖励（确保格式正确）
    ]
    
    trainer = GRPOTrainer(
        model=policy_model,              # 策略模型
        reward_funcs=reward_functions,   # 奖励函数组合
        args=training_args,              # 训练参数
        train_dataset=train_dataset,     # 训练数据集
        processing_class=tokenizer,      # 分词器
        peft_config=peft_config          # LoRA配置
    )
    
    print("GRPO训练器创建完成")
    
    # ====================== 3.10 执行训练 ======================
    print("开始GRPO强化学习训练...")
    print("=" * 50)
    
    """
    GRPO训练过程：
        1. 对每个prompt，生成num_generations个回答
        2. 使用奖励函数组合评估每个回答的质量
        3. 在组内计算相对优势（回答之间的相对表现）
        4. 使用KL散度约束更新策略模型
        5. 重复直到收敛
    """
    trainer.train()
    
    print("=" * 50)
    print("GRPO训练完成!")
    
    # ====================== 3.11 保存训练好的模型 ======================
    # 保存LoRA权重和训练配置
    trainer.save_model(training_args.output_dir)
    print(f"训练好的模型已保存到: {training_args.output_dir}")
    
    # ====================== 3.12 训练后推理测试 ======================
    """
    训练后评估：验证GRPO训练效果
    通过多样化的测试问题，检验模型的泛化能力
    """
    
    print("\n" + "=" * 60)
    print("开始推理测试 - 验证训练效果")
    print("=" * 60)
    
    # 加载训练好的模型（带LoRA权重）
    trained_model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    print("训练好的模型加载完成")
    
    # 配置生成参数
    generation_config = GenerationConfig(
        max_new_tokens=256,               # 最大生成token数
        temperature=0.7,                  # 温度参数（平衡创造性和确定性）
        top_p=0.9,                        # 核采样
        top_k=20,                         # Top-K采样
        do_sample=True,                   # 启用采样
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 测试问题（包含训练时未见过的prompt）
    test_questions = [
        "请用优美的话语夸赞和鼓励我：",  # 创意写作类
        "请用夸张的话语夸赞我：",       # 风格变化类
        "请用搞笑的话语夸赞我："        # 幽默类
    ]
    
    print(f"\n准备测试 {len(test_questions)} 个问题...")
    
    for i, question in enumerate(test_questions, 1):
        # 构建Qwen对话格式
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(trained_model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = trained_model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True
            )
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        
        # 提取assistant的回复部分
        if "<|im_start|>assistant" in generated_text:
            response = generated_text.split("<|im_start|>assistant")[-1]
            response = response.replace("<|im_end|>", "").strip()
        else:
            response = generated_text.replace(prompt, "").strip()
        
        # 输出测试结果
        print(f"\n测试 {i}:")
        print(f"问题: {question}")
        print(f"回答: {response}")
        print("-" * 50)
    
    # ====================== 3.13 测试完成总结 ======================
    print("\n" + "=" * 60)
    print("推理测试完成！")
    print("=" * 60)
    print("\nGRPO训练和推理测试全部完成！")
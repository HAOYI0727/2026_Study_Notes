"""
Name: Qwen3_PPO.py
Description: 基于Qwen3-4B-Instruct模型的PPO(Proximal Policy Optimization)强化学习训练实现
             包含4-bit量化、LoRA参数高效微调、预训练奖励模型加载、多模型协同训练
"""

"""
PPO强化学习在RLHF中的核心作用：
    1. 策略优化：通过奖励信号优化语言模型的生成策略
    2. 价值评估：使用价值模型估计状态价值，减少方差
    3. 信任区域：通过KL散度约束防止策略更新过快
    4. 奖励建模：使用预训练奖励模型提供细粒度的质量评估
"""

# ====================== 1. 镜像加速与导入必要库 ======================

import os
import shutil
import torch
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息，保持输出整洁

# 设置国内镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 核心机器学习库
from accelerate import PartialState  # 分布式训练状态管理
from datasets import Dataset         # 数据集处理
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    GenerationConfig
)

# TRL库相关导入 - PPO是RLHF的核心组件
from trl import ModelConfig, get_peft_config
from trl.experimental.ppo import PPOConfig, PPOTrainer

if __name__ == "__main__":
    # ====================== 2. 参数配置定义 ======================
    
    # ====================== 2.1 PPO训练参数 ======================
    """
    PPOConfig参数: 继承自TrainingArguments，添加了PPO特有的训练参数
    """
    training_args = PPOConfig(
        # 基础训练参数
        learning_rate=3e-6,                  # 学习率：控制参数更新步长，较小的值适合微调
        output_dir="./Qwen3_PPO_Reward_Model",  # 输出目录：训练好的模型保存位置
        per_device_train_batch_size=1,       # 批次大小：每设备每次处理的样本数
        gradient_accumulation_steps=4,       # 梯度累积：模拟更大的批次训练
        total_episodes=10,                   # 训练轮数：适合演示的小规模训练
        
        # PPO特有参数
        missing_eos_penalty=1.0,             # EOS惩罚：对缺失结束标记的惩罚系数,鼓励模型生成完整回答
        
        # 数据处理和日志
        dataset_num_proc=1,                  # 数据处理进程数
        report_to="none"                     # 报告平台：不向wandb等平台报告
    )
    
    # ====================== 2.2 模型配置参数 ======================
    """
    ModelConfig参数：封装了模型加载和LoRA配置的参数
    """
    model_args = ModelConfig(
        model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",  # 基础模型：Qwen3-4B指令微调版本
        load_in_4bit=True,                                 # 4-bit量化：大幅减少显存占用
        trust_remote_code=True,                            # 信任远程代码：Qwen模型需要此设置
        
        # LoRA配置参数
        lora_r=8,                                          # LoRA秩：控制适配器参数数量
        lora_alpha=16,                                     # LoRA缩放系数：影响适配器权重
        lora_dropout=0.05,                                 # LoRA dropout：防止过拟合
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                             "gate_proj", "down_proj", "up_proj"]     # LoRA目标模块：适配Transformer的关键层
    )
    
    # ====================== 2.4 清理输出目录 ======================
    # 删除之前的训练结果，确保干净的训练环境
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    print(f"已清理输出目录: {training_args.output_dir}")

    # ====================== 3. 模型和分词器加载 ======================
    
    """
    PPO算法核心组件说明：
        1. 策略模型(Policy Model): 要优化的语言模型，负责生成回答
        2. 奖励模型(Reward Model): 评估生成质量，提供奖励信号
        3. 价值模型(Value Model): 估计状态价值，用于优势计算
        4. 参考模型(Reference Model): 原始模型，用于KL散度约束
    """
    # ====================== 3.1 量化配置 ======================
    """
    4-bit量化配置
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # 启用4-bit量化
        bnb_4bit_use_double_quant=True,         # 使用双量化进一步压缩
        bnb_4bit_quant_type="nf4",              # 量化类型：Normal Float 4
        bnb_4bit_compute_dtype=torch.bfloat16   # 计算精度：bfloat16
    )
    
    # ====================== 3.2 分词器加载 ======================
    # 加载Qwen模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",   
        trust_remote_code=model_args.trust_remote_code
    )
    
    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("分词器加载完成")
    print(f"  - pad_token: {tokenizer.pad_token}")
    print(f"  - vocab_size: {len(tokenizer)}")
    
    # ====================== 3.3 策略模型加载 ======================
    """
    策略模型(Policy Model): 负责生成文本回答，通过PPO算法优化生成策略，使用4-bit量化加载，减少显存
    """
    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,  # 应用4-bit量化
        device_map="auto",                        # 自动设备映射
        trust_remote_code=True,                   # 信任远程代码
        dtype=torch.bfloat16                      # 使用bfloat16精度
    )
    
    print("策略模型加载完成")
    print(f"  - 模型类型: {type(policy).__name__}")
    print(f"  - 参数量: ~4B")
    
    # ====================== 3.4 奖励模型加载 ======================
    """
    奖励模型(Reward Model) ：从本地加载已训练好的奖励模型，提供更准确的奖励信号，替代随机初始化的奖励模型
    奖励模型的作用：评估生成回答的质量、输出标量奖励分数、指导策略模型优化
    """
    # 加载已训练好的奖励模型
    reward_model_path = "./Qwen_PPO_Reward_Model"
    
    # 检查奖励模型文件夹是否存在
    if os.path.exists(reward_model_path):
        print(f"加载已训练好的奖励模型: {reward_model_path}")
        # 从本地文件夹加载训练好的奖励模型
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            num_labels=1,                            # 单输出：奖励分数
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
        print(f"✓ 成功加载预训练奖励模型")
    else:
        print("警告：未找到训练好的奖励模型，将重新初始化奖励模型")
        # 如果找不到训练好的模型，回退到重新初始化
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,                            # 单输出：奖励分数
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
        print("✗ 使用随机初始化的奖励模型（效果可能较差）")
    
    print("奖励模型加载完成")
    
    # ====================== 3.5 价值模型加载 ======================
    """
    价值模型(Value Model): 估计状态价值函数，用于计算优势函数，减少方差，辅助PPO算法稳定训练
    """
    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,                            # 单输出：状态价值
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    print("价值模型加载完成")
    
    # ====================== 3.6 LoRA配置 ======================
    # 获取LoRA微调配置（从model_args转换）
    peft_config = get_peft_config(model_args)
    ref_policy = None  # 不使用参考模型（简化版本）
    
    print(f"LoRA配置完成")
    print(f"  - lora_r: {model_args.lora_r}")
    print(f"  - target_modules: {model_args.lora_target_modules}")

    # ====================== 4. 数据集准备 ======================
    
    # ====================== 4.1 创建示例查询数据 ======================
    """
    数据集格式要求：每个样本包含"query"字段、使用Qwen的ChatML格式、查询部分以assistant标记结束，等待模型生成
    """
    # 构建Qwen对话格式的示例查询，用于PPO训练
    queries = [
        {"query": "<|im_start|>user\n什么是LoRA？<|im_end|>\n<|im_start|>assistant\n"},
        {"query": "<|im_start|>user\nLoRA有什么优势？<|im_end|>\n<|im_start|>assistant\n"},
        {"query": "<|im_start|>user\n如何使用LoRA进行模型微调？<|im_end|>\n<|im_start|>assistant\n"},
        {"query": "<|im_start|>user\n什么是PPO算法？<|im_end|>\n<|im_start|>assistant\n"},
        {"query": "<|im_start|>user\nRLHF的全称是什么？<|im_end|>\n<|im_start|>assistant\n"}
    ]
    
    # 转换为HuggingFace Dataset格式
    train_dataset = Dataset.from_list(queries)
    eval_dataset = train_dataset  # 评估数据集与训练数据集相同
    dataset_text_field = "query"  # 数据集中的文本字段名
    
    print(f"创建了 {len(queries)} 个训练样本")
    for i, q in enumerate(queries):
        print(f"  样本 {i+1}: {q['query'][:50]}...")
    
    # ====================== 4.2 数据集预处理函数 ======================
    def prepare_dataset(dataset, tokenizer):
        """
        数据集预处理函数：对文本进行分词处理
        PPO训练需要的格式：
            - input_ids: 分词后的token序列
            - attention_mask: 注意力掩码（由训练器自动添加）
        参数:dataset: 原始数据集
            tokenizer: 分词器对象
        返回:分词后的数据集
        """
        def tokenize(element):
            """分词函数：将文本转换为input_ids"""
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,    # 不进行填充，训练时动态填充
                truncation=True,
                max_length=512
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,    # 移除原始列，只保留input_ids
            num_proc=training_args.dataset_num_proc,
        )

    # ====================== 4.3 执行数据预处理 ======================
    # 仅在主进程上执行数据预处理（分布式训练时避免重复处理）
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
    
    print("数据集预处理完成")
    print(f"  - 训练样本数: {len(train_dataset)}")

    # ====================== 5. 训练器创建和训练执行 ======================
    
    # ====================== 5.1 创建PPO训练器 ======================
    """
    PPOTrainer初始化：
        1. 策略模型：要优化的生成模型
        2. 奖励模型：提供奖励信号（预训练）
        3. 价值模型：估计状态价值
        4. 训练配置：PPO超参数
        5. 数据集：预处理后的训练数据
        6. LoRA配置：参数高效微调
    """
    trainer = PPOTrainer(
        args=training_args,           # PPO训练参数
        processing_class=tokenizer,   # 分词器，用于文本处理
        model=policy,                 # 策略模型（生成模型）
        ref_model=ref_policy,         # 参考模型（可选，用于KL散度惩罚）
        reward_model=reward_model,    # 奖励模型（评估生成质量）- 预训练版本
        value_model=value_model,      # 价值模型（估计状态价值）
        train_dataset=train_dataset,  # 训练数据集
        eval_dataset=eval_dataset,    # 评估数据集
        peft_config=peft_config,      # LoRA微调配置
    )
    
    print("PPO训练器创建完成")
    print(f"  - 训练设备: {trainer.accelerator.device}")
    print(f"  - 是否分布式: {trainer.accelerator.distributed_type}")
    
    # ====================== 5.2 执行训练 ======================
    print("开始PPO强化学习训练...")
    print("=" * 50)
    
    """
    PPO训练循环：
        1. 对每个prompt，策略模型生成回答
        2. 奖励模型评估回答质量
        3. 价值模型估计状态价值
        4. 计算优势函数
        5. 更新策略模型（带KL约束）
        6. 更新价值模型
    """
    trainer.train()
    
    print("=" * 50)
    print("PPO训练完成!")
    
    # ====================== 5.3 保存训练好的模型 ======================
    # 保存LoRA权重和训练配置
    trainer.save_model(training_args.output_dir)
    print(f"训练好的模型已保存到: {training_args.output_dir}")

    # ====================== 6. 训练后推理测试 ======================
    
    print("\n" + "=" * 60)
    print("开始推理测试 - 验证训练效果")
    print("=" * 60)
    
    # ====================== 6.1 加载训练好的模型 ======================
    # 从保存的目录加载训练好的策略模型
    trained_policy = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    print("训练好的模型加载完成")
    
    # ====================== 6.2 配置生成参数 ======================
    
    generation_config = GenerationConfig(
        max_new_tokens=256,           # 最大生成长度
        temperature=0.7,              # 温度参数：控制随机性
        top_p=0.9,                    # 核采样参数：控制多样性
        top_k=20,                     # Top-k采样：Qwen模型默认值
        do_sample=True,               # 启用采样生成
        pad_token_id=tokenizer.eos_token_id,  # 填充标记
        eos_token_id=tokenizer.eos_token_id,  # 结束标记
        bos_token_id=151643           # 开始标记：Qwen模型默认值
    )
    print("生成参数配置完成")
    
    # ====================== 6.3 定义测试问题 ======================
    # 测试不同风格的夸赞请求，验证模型的理解和生成能力
    test_questions = [
        "请用优美的话语夸赞和鼓励我：",
        "请用夸张的话语夸赞我：",
        "请用搞笑的话语夸赞我："
    ]
    
    print(f"\n准备测试 {len(test_questions)} 个问题...")
    
    # ====================== 6.4 执行推理测试 ======================
    print("\n" + "=" * 60)
    print("推理测试结果")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        # 构建Qwen对话格式的提示词
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码输入文本
        inputs = tokenizer(prompt, return_tensors="pt").to(trained_policy.device)
        
        # 使用训练好的模型生成回复
        with torch.no_grad():
            outputs = trained_policy.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True
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
    
    # ====================== 6.5 测试完成总结 ======================
    print("\n" + "=" * 60)
    print("推理测试完成！")
    print("=" * 60)
    print("\n脚本执行完毕，PPO训练和推理测试全部完成！")
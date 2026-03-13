"""
Name: Qwen3_GRPO_Reward_Model.py
Description: 基于Qwen3-4B-Instruct模型的奖励模型(Reward Model)训练实现，用于PPO/GRPO强化学习中的奖励信号提供，通过偏好数据学习人类偏好
"""

"""
奖励模型的核心原理：
    1. 输入：一对文本（chosen: 偏好回答, rejected: 非偏好回答）
    2. 输出：标量奖励分数，表示回答的质量
    3. 训练目标：使chosen回答的分数高于rejected回答的分数
    4. 应用场景：为强化学习提供细粒度的奖励信号，替代人工反馈
"""

# ====================== 1. 镜像加速与导入必要库 ======================

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

import torch
from datasets import Dataset  # HuggingFace数据集库
from transformers import (
    AutoModelForSequenceClassification,  # 用于序列分类任务的预训练模型,奖励模型本质是一个二分类器（偏好/非偏好），输出一个标量分数
    AutoTokenizer,  # 自动加载与预训练模型匹配的Tokenizer
    BitsAndBytesConfig,  # 配置模型量化参数的类
)

# 导入LoRA相关库（参数高效微调）
from peft import (
    LoraConfig,      # LoRA配置类，用于定义LoRA的参数（如秩、alpha值、目标模块等）
    get_peft_model,  # 创建PeftModel实例的函数，将基础模型与LoRA配置结合
)

"""
RewardTrainer vs 普通Trainer的优势：
    1. 内置偏好数据处理：自动处理chosen/rejected配对
    2. 专用损失函数：使用ranking loss（如Pairwise Ranking Loss）
    3. 准确率监控：自动计算chosen分数高于rejected的比例
    4. 边界控制：确保chosen和rejected的分数有足够差距
"""
# 导入RewardTrainer和RewardConfig（TRL库专门用于奖励模型训练的组件）
from trl import (
    RewardTrainer,   # 专门为奖励模型训练优化的训练器
    RewardConfig     # 奖励模型训练配置
)

# ====================== 2. 模型配置 ======================

model_name = "Qwen/Qwen3-4B-Instruct-2507"  # 基础模型

# 4-bit量化配置（大幅降低显存占用）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 启用4-bit量化加载
    bnb_4bit_use_double_quant=True,       # 使用双重量化（对量化常数再次量化），进一步节省显存
    bnb_4bit_quant_type="nf4",            # 使用NF4量化类型，专门为神经网络权重优化的量化格式
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算时使用bfloat16精度，平衡精度和效率
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,   # Qwen使用自定义代码
    padding_side="right"      # 因果语言模型需要右侧padding
)

# 设置pad_token（Qwen模型通常没有默认pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

"""
AutoModelForSequenceClassification与AutoModelForCausalLM的区别：
    - AutoModelForCausalLM：用于生成任务，输出logits形状为[batch, seq_len, vocab_size]
    - AutoModelForSequenceClassification：用于分类任务，添加了分类头，输出logits形状为[batch, num_labels]
    奖励模型需要输出一个标量分数，所以设置num_labels=1
"""
# 加载序列分类模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,       # 应用量化配置
    device_map="auto",                    # 自动设备映射
    dtype=torch.bfloat16,                 # 使用bfloat16精度
    trust_remote_code=True,               # 允许加载自定义代码
    num_labels=1                          # 奖励模型输出一个标量值
)

# 设置pad_token_id，确保模型正确处理padding
model.config.pad_token_id = tokenizer.pad_token_id

# ====================== 3. LoRA配置 ======================

"""
奖励模型的LoRA配置：与SFT类似，但任务类型变为SEQ_CLS（序列分类），选择全量模块进行微调，以获得更好的奖励建模能力
"""
peft_config = LoraConfig(
    r=8,               # LoRA秩，控制可训练参数数量
    target_modules=[   # 要应用LoRA的目标模块（全量选择）
        "q_proj",      # 注意力查询投影
        "v_proj",      # 注意力值投影
        "k_proj",      # 注意力键投影
        "o_proj",      # 注意力输出投影
        "gate_proj",   # MLP门控投影
        "down_proj",   # MLP下降投影
        "up_proj"      # MLP上升投影
    ],
    task_type="SEQ_CLS",  # 任务类型：序列分类（与因果语言模型不同）
    lora_alpha=16,        # LoRA缩放参数
    lora_dropout=0.05,    # Dropout比率
    bias="none"           # 不训练偏置
)

"""
prepare_model_for_kbit_training函数作用：
主要功能：
    1. 启用梯度检查点（gradient checkpointing）以节省显存
    2. 处理4-bit量化模型的线性层，确保梯度正确计算
    3. 配置输入数据类型，确保与量化参数兼容
    4. 为LoRA训练做必要的预处理
注意事项：
    如果遇到梯度计算错误或训练不稳定的问题，可以重新启用此函数
"""
# model = prepare_model_for_kbit_training(model)  # 已注释，非必需

# 创建LoRA模型
peft_model = get_peft_model(model, peft_config)  # 冻结原始模型参数，只在指定模块插入可训练的LoRA参数
peft_model.print_trainable_parameters()  # 查看可训练参数统计信息

# ====================== 4. 数据准备 ======================

"""
偏好数据集格式：
RewardTrainer要求的数据格式：
    - chosen: 被偏好的回答（高质量回答）
    - rejected: 不被偏好的回答（低质量回答）
数据格式要求：
    1. 必须包含完整的对话上下文（包括user和assistant部分）
    2. 使用模型特定的对话模板（这里使用Qwen的ChatML格式）
    3. chosen和rejected应该是同一问题的不同回答
训练原理：
    - 模型需要学会给chosen回答打高分，给rejected回答打低分
    - 通过ranking loss优化，确保分数差距足够大
"""
sample_data = [
    {
        "chosen": "<|im_start|>user\n什么是LoRA？<|im_end|>\n<|im_start|>assistant\nLoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数，同时保持较好的微调效果。它能够大幅降低显存需求，适用于大规模语言模型的微调。<|im_end|>",
        "rejected": "<|im_start|>user\n什么是LoRA？<|im_end|>\n<|im_start|>assistant\nLoRA是一个公司的名字，主要生产电子产品。<|im_end|>"
    },
    {
        "chosen": "<|im_start|>user\nLoRA有什么优势？<|im_end|>\n<|im_start|>assistant\nLoRA的主要优势包括：1) 大幅减少训练参数；2) 降低显存需求；3) 保持较好的微调效果；4) 支持模型多任务微调；5) 可以与其他参数高效微调方法结合使用。<|im_end|>",
        "rejected": "<|im_start|>user\nLoRA有什么优势？<|im_end|>\n<|im_start|>assistant\nLoRA的优势是它的名字很好听，容易记住。<|im_end|>"
    },
    {
        "chosen": "<|im_start|>user\n如何使用LoRA进行模型微调？<|im_end|>\n<|im_start|>assistant\n使用LoRA进行模型微调的步骤包括：1) 准备训练数据；2) 配置LoRA参数（如秩、目标模块等）；3) 加载预训练模型并应用LoRA配置；4) 设置训练超参数；5) 开始训练；6) 保存和加载微调后的模型。<|im_end|>",
        "rejected": "<|im_start|>user\n如何使用LoRA进行模型微调？<|im_end|>\n<|im_start|>assistant\n使用LoRA很简单，只需要点击一个按钮就可以了。<|im_end|>"
    }
]

# 转换为Dataset格式
tokenized_dataset = Dataset.from_list(sample_data)

# ====================== 5. 训练配置 ======================

reward_config = RewardConfig(
    output_dir="./Qwen3_GRPO_Reward_Model",         # 模型和检查点的输出目录
    per_device_train_batch_size=1,             # 每个设备的训练批次大小
    gradient_accumulation_steps=4,             # 梯度累积步数（实际batch_size=4）
    learning_rate=2e-4,                        # 学习率（奖励模型通常用稍大的学习率）
    num_train_epochs=3,                        # 训练轮数
    bf16=True,                                 # 启用BF16混合精度训练
    optim="adamw_bnb_8bit",                    # 使用8-bit量化的AdamW优化器
    lr_scheduler_type="cosine",                # 余弦退火学习率调度器
    warmup_ratio=0.05,                         # 学习率预热比例
    report_to="none",                          # 禁用外部日志报告
    # 奖励模型特有参数（通过RewardConfig自动配置）
    # - margin: 自动计算chosen和rejected的分数差
    # - accuracy: 自动监控chosen分数高于rejected的比例
)

# ====================== 6. 训练模型 ======================

"""
RewardTrainer的工作流程：
    1. 对每个样本的chosen和rejected进行分词
    2. 分别计算两个回答的奖励分数
    3. 计算ranking loss，确保chosen分数 > rejected分数
    4. 监控准确率（chosen > rejected的比例）和边界（分数差）
    5. 反向传播优化LoRA参数
"""
trainer = RewardTrainer(
    model=peft_model,                    # 要训练的模型（LoRA模型）
    args=reward_config,                  # 训练配置参数
    train_dataset=tokenized_dataset,     # 训练数据集
    processing_class=tokenizer           # 分词器（用于数据预处理）
)

# 开始训练
print("开始训练奖励模型...")
trainer.train()
print("训练完成！")

# ====================== 7. 测试模型 ======================

def test_reward_model():
    """
    测试训练好的奖励模型，验证模型是否能正确区分好回答和差回答
    """
    print("\n测试奖励模型...")
    
    # 测试问题
    question = "什么是LoRA？"
    # 好的回答和差的回答（与训练数据类似但略有变化）
    good_response = "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数，同时保持较好的微调效果。"
    bad_response = "LoRA是一个公司的名字，主要生产电子产品。"
    
    # 格式化输入（使用Qwen的ChatML格式）
    good_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{good_response}<|im_end|>"
    bad_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{bad_response}<|im_end|>"
    
    # 分词处理
    good_inputs = tokenizer(
                    good_prompt, 
                    return_tensors="pt",   # 返回PyTorch张量格式
                    truncation=True,       # 超过max_length的文本会被截断
                    max_length=1024        # 设置最大序列长度
                )
    bad_inputs = tokenizer(bad_prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    # 移动到模型所在设备
    good_inputs = {k: v.to(peft_model.device) for k, v in good_inputs.items()}
    bad_inputs = {k: v.to(peft_model.device) for k, v in bad_inputs.items()}
    
    # 获取奖励分数（禁用梯度计算）
    with torch.no_grad():
        good_score = peft_model(**good_inputs).logits.item()  # 提取标量分数
        bad_score = peft_model(**bad_inputs).logits.item()
    
    # 输出测试结果
    print(f"问题: {question}")
    print(f"好的回答: {good_response[:50]}...") 
    print(f"好的回答分数: {good_score:.4f}")
    print(f"差的回答: {bad_response}")
    print(f"差的回答分数: {bad_score:.4f}")
    print(f"分数差: {good_score - bad_score:.4f}")
    
    # 验证模型效果
    if good_score > bad_score:
        print("✓ 模型正确区分了好坏回答")
    else:
        print("✗ 模型未能正确区分好坏回答")

# 执行测试
test_reward_model()

# ====================== 8. 保存模型 ======================

save_path="./Qwen3_GRPO_Reward_Model"
# 保存LoRA模型权重（只保存可训练参数）
peft_model.save_pretrained(save_path)
# 保存分词器配置
tokenizer.save_pretrained(save_path)
print(f"\n模型已保存到{save_path}")
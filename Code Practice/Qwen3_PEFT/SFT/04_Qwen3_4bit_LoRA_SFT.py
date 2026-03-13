"""
Name: Qwen3_4bit_LoRA_SFT.py
Description: 基于Qwen3-4B-Instruct模型的4-bit量化 + LoRA指令微调实现，使用transformers库的Trainer高级API进行训练，大幅降低显存占用并简化训练流程。
"""

# ====================== 1. 镜像加速与导入必要库 ======================

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

import torch
from datasets import Dataset  # HuggingFace数据集库，提供高效的数据处理接口
from transformers import (
    AutoModelForCausalLM,  # 自动加载适用于因果语言建模（如文本生成）的预训练模型，会根据模型名称自动选择合适的模型架构
    AutoTokenizer,  # 自动加载与预训练模型匹配的Tokenizer（分词器）
                    # 将自然语言文本转换为模型可识别的数字token（词表索引），同时处理文本截断、填充等操作
    BitsAndBytesConfig,  # 配置模型量化参数，通过4-bit/8-bit量化大幅降低模型显存占用，使大模型能够在消费级GPU上运行
    TrainingArguments,  # 配置模型训练过程中所有超参数和训练设置
                        # 包含训练轮数、学习率、批处理大小、保存路径、日志频率等关键参数，是控制训练流程的核心配置，直接影响训练效果和效率
    Trainer,  # transformers库提供的高层训练器类
              # 封装了完整的训练循环（前向传播、损失计算、反向传播、参数更新）、验证流程、模型保存、日志记录等功能
    DataCollatorForLanguageModeling  # 因果语言建模任务的数据拼接器
                                     # 将批量数据中的文本样本动态padding成统一长度，并生成对应的标签，确保输入数据格式符合模型训练要求，同时避免固定padding造成的计算浪费
)
'''
PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）是针对大模型微调的优化技术，
通过仅训练模型的一小部分参数（而非全部参数），在降低内存占用和计算成本的同时，保持较好的微调效果。LoRA是PEFT中最常用的方法之一。
'''
from peft import (
    LoraConfig,  # 配置LoRA（Low-Rank Adaptation，低秩适配）微调参数
                 # 关键参数包括秩（rank）、缩放因子（lora_alpha）、目标模块（target_modules）等，这些参数决定了微调的参数量和学习能力
    get_peft_model  # 根据基础模型和LoRA配置，生成带有LoRA适配器的PEFT模型
                    # 该函数会在基础模型的指定层（如Transformer的注意力层）插入LoRA参数，并冻结基础模型的大部分参数，仅保留LoRA参数可训练
)

# ====================== 2. 模型配置 ======================

model_name = "Qwen/Qwen3-4B-Instruct-2507"  # 指定Qwen3-4B指令微调版本

# 4-bit量化配置（大幅降低显存占用的核心技术）
# 量化原理：将模型权重从16位浮点数压缩到4位整数，通过牺牲少量精度换取显存节省
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 启用4-bit量化加载
    bnb_4bit_use_double_quant=True,        # 使用双重量化（对量化常数再次量化），进一步节省显存
    bnb_4bit_quant_type="nf4",             # 使用NF4（4-bit NormalFloat）量化类型，专门为神经网络权重优化的量化格式
    bnb_4bit_compute_dtype=torch.bfloat16  # 计算时使用bfloat16精度，平衡精度和效率
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,  # 允许加载远程代码（因为Qwen使用了自定义模型结构），加载Qwen等非Hugging Face官方标准架构的模型时需要
    padding_side="right"     # 设置padding在右侧，因果语言模型通常需要右侧padding，从右侧开始填充，确保因果掩码正确
)

# 设置pad_token（Qwen模型通常没有默认的pad_token，需要手动设置）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token           # 使用eos_token作为pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载基础模型（带4-bit量化）
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,   # 应用已有的量化配置
    device_map="auto",                # 自动分配模型到可用设备（GPU/CPU）
    dtype=torch.bfloat16,             # 模型权重以及后续计算使用bfloat16精度
    trust_remote_code=True            # 允许加载自定义代码
)

# 训练前配置
base_model.config.use_cache = False   # 训练期间彻底禁用KV Cache

"""
模型结构探索：
# 在加载 base_model 后，查看模型的所有层结构，帮助理解模型架构，这对于确定LoRA要应用的目标模块非常有用
for name, module in base_model.named_modules():
    print(name)
    print(module)
    print('-------------------------------------')
# 打印完成后直接退出程序
import sys
sys.exit()
"""

# ====================== 3. LoRA配置（核心） ======================

# LoRA原理：在原始权重矩阵旁添加低秩可训练矩阵（A和B），原始权重保持不变
# 前向传播变为 h = Wx + BAx，其中B和A是可训练的低秩矩阵
peft_config = LoraConfig(
    r=8,               # LoRA秩（rank），控制可训练参数数量，r越小参数越少，通常设为8-64之间
    target_modules=[   # 要应用LoRA的目标模块，这些名称来自打印的模型结构
        "q_proj",      # 注意力查询投影
        "v_proj",      # 注意力值投影
        "k_proj",      # 注意力键投影
        "o_proj",      # 注意力输出投影
        "gate_proj",   # MLP门控投影（SwiGLU结构）
        "down_proj",   # MLP下降投影
        "up_proj"      # MLP上升投影
    ],
    task_type="CAUSAL_LM",  # 任务类型：因果语言模型
    lora_alpha=16,          # LoRA缩放参数，控制低秩矩阵的权重，通常设置为r的2倍
    lora_dropout=0.05,      # LoRA层的dropout比率，防止过拟合
    bias="none"             # 是否训练偏置项，通常保持none以节省参数
)

# 创建LoRA模型

peft_model = get_peft_model(base_model, peft_config)  # 冻结原始模型参数，只在指定模块插入可训练的LoRA参数
peft_model.print_trainable_parameters()  # 查看可训练参数统计信息

# ====================== 4. 数据准备 ======================

# 简单的训练数据
sample_data = [
    {
        "instruction": "什么是LoRA？",
        "response": "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数。"
    },
    {
        "instruction": "LoRA有什么优势？",
        "response": "LoRA可以大幅减少训练参数，降低显存需求，同时保持较好的微调效果。"
    }
]

# 将数据转换为HuggingFace Dataset格式
dataset = Dataset.from_list(sample_data)

# 数据处理函数：将原始文本转换为模型输入格式
def process_data(example):
    # 格式化为Qwen要求的prompt格式（ChatML格式）
    # Qwen使用特殊的标记来区分不同角色：
    # <|im_start|>user 标记用户输入开始
    # <|im_end|> 标记对话结束
    # <|im_start|>assistant 标记助手回答开始
    prompt = f"""<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['response']}<|im_end|>"""
    
    # 设置分词处理
    encoding = tokenizer(
        prompt,
        truncation=True,        # 超过max_length的文本会被截断
        max_length=1024,        # 设置最大序列长度
        return_tensors="pt"     # 返回PyTorch张量格式
    )
    
    # 设置labels：在因果语言模型中，labels通常与input_ids相同
    # Trainer会在内部自动处理label的偏移（只计算预测部分的损失）
    encoding["labels"] = encoding["input_ids"].clone()
    
    # 返回处理后的数据，flatten()将张量展平为一维
    return {
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),  # 填充掩码，标记哪些是真实token，哪些是padding
        "labels": encoding["labels"].flatten()
    }

# 应用数据处理函数到整个数据集
tokenized_dataset = dataset.map(process_data, remove_columns=dataset.column_names)  # 移除原始列，只保留处理后的列

# 数据整理器（Data Collator）的核心作用：
# 1. 动态padding：在每个训练批次中，将样本填充到该批次内最长样本的长度，相比固定padding到最大长度（如1024），动态padding能显著减少不必要的计算
# 2. 自动生成attention mask：根据padding情况生成对应的attention mask
# 3. 处理标签：对于因果语言建模，自动处理标签偏移
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 采用因果语言建模（Causal Language Modeling），而不是掩码语言建模（MLM）
               # 表示使用自回归方式，即每个token只能看到前面的token
)

# ====================== 5. 训练配置 ======================

# TrainingArguments包含了所有训练相关的超参数设置
training_args = TrainingArguments(
    output_dir="./Qwen3_4bit_LoRA_SFT_checkpoint",      # 模型和检查点的输出目录
    per_device_train_batch_size=1,         # 每个设备（如GPU）的训练批次大小
    gradient_accumulation_steps=4,         # 梯度累积步数：通过累积多个小批次的梯度实现大批次效果
                                           # 实际批次大小 = per_device_train_batch_size * gradient_accumulation_steps
    learning_rate=2e-4,                    # 优化器的初始学习率（LoRA通常使用较大的学习率）
    num_train_epochs=2,                    # 训练总轮数
    fp16=True,                             # 启用FP16混合精度训练，加速训练并减少显存
    optim="adamw_bnb_8bit",                # 使用8-bit量化的AdamW优化器，大幅减少优化器显存占用，可以节省大量显存
    lr_scheduler_type="cosine",            # 学习率调度器类型，使用余弦退火策略，学习率先缓慢下降，然后加速下降，最后趋于平稳
    warmup_ratio=0.05,                     # 学习率预热的比例，前5%的训练步数从0线性增加到初始学习率，有助于稳定训练初期
    report_to="none"                       # 禁用外部日志报告（如W&B、TensorBoard），仅本地输出
)

# ====================== 6. 训练模型 ======================

# Trainer封装了完整的训练循环，简化训练代码
trainer = Trainer(
    model=peft_model,                  # 要训练的模型（LoRA模型）
    args=training_args,                # 训练参数配置
    train_dataset=tokenized_dataset,   # 训练数据集
    data_collator=data_collator        # 数据整理器，负责批处理
)

# 开始训练
print("开始训练...")
trainer.train()  # 执行训练循环
print("训练完成！")

# ====================== 7. 推理测试 ======================

def generate_answer(instruction):
    """
    使用训练好的模型生成回答
    参数：instruction: 用户输入的指令文本
    返回：模型生成的回答文本
    """
    # 构造符合Qwen格式的对话消息
    messages = [{"role": "user", "content": instruction}]
    
    # 应用聊天模板生成prompt
    prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,   # 在末尾添加助手标记，指示模型开始生成
                tokenize=False                # 返回文本而不是token IDs，便于查看
            )
    # 分词并移动到模型所在设备
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
    
    # 生成回答（禁用梯度计算以节省显存）
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_new_tokens=100,      # 最大生成新token数
            temperature=0.7          # 温度参数，控制随机性
        )
    
    # 解码生成的token序列
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取助手回答部分（去除prompt部分）
    answer = answer.split("<|im_start|>assistant\n")[-1]
    return answer

# 测试推理
print("\n推理测试：")
test_question = "请不留余地的夸赞和鼓励我"
print(f"我的询问：{test_question}")
print(f"AI的回答：{generate_answer(test_question)}")

# ====================== 8. 保存模型 ======================

save_path="./04_Qwen3_4bit_LoRA_SFT_model"
# 保存LoRA模型权重（只保存可训练参数，文件很小）
peft_model.save_pretrained(save_path)
# 保存分词器配置
tokenizer.save_pretrained(save_path)
print(f"\n模型已保存到{save_path}")

"""
核心技术：
   - 4-bit量化：通过BitsAndBytesConfig实现模型量化，将模型显存占用降低约75%
   - LoRA微调：仅训练0.4%的参数，进一步降低显存需求
   - Trainer封装：使用高级API简化训练流程，减少代码量
   - 动态padding：通过DataCollator实现动态批次填充，提高训练效率

显存优化策略：
   - 4-bit量化：将模型权重从16位压缩到4位
   - LoRA：只训练少量新增参数
   - 梯度累积：通过累积多个小批次实现大批次效果
   - 8-bit优化器：使用8-bit AdamW减少优化器显存占用
   - FP16混合精度：加速训练并减少显存

使用方法：
   - 准备数据：按instruction-response格式准备训练数据
   - 调整参数：根据显存和任务需求调整LoRA参数（r、target_modules）
   - 训练监控：观察loss下降情况，判断训练效果
   - 推理测试：使用generate_answer函数测试模型效果
"""
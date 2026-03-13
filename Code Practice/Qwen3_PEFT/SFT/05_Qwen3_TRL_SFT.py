"""
Name: Qwen3_TRL_SFT.py
Description: 基于Qwen3-4B-Instruct模型的LoRA指令微调实现，使用TRL库的SFTTrainer高级API，
      采用prompt-completion数据格式，利用completion_only_loss自动实现损失掩码，并启用NEFTune噪声训练增强泛化能力。
"""

# ====================== 1. 镜像加速与导入必要库 ======================

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置HuggingFace镜像源加速国内下载

import torch
from datasets import Dataset  # HuggingFace数据集库，提供高效的数据处理接口
from transformers import (
    AutoModelForCausalLM,    # 自动加载因果语言模型
    AutoTokenizer,           # 自动加载分词器
    BitsAndBytesConfig,      # 量化配置
)
from peft import (
    LoraConfig,              # LoRA配置类
    get_peft_model           # 获取PEFT模型
)
# 导入TRL库的SFT相关组件
# TRL (Transformer Reinforcement Learning) 是HuggingFace开发的专门用于大模型训练的高级库
"""
TRL库的优势：
1. 简化代码：无需手动实现损失计算、数据整理等底层逻辑
2. 内置优化：自动处理padding、attention mask、标签偏移等问题
3. 支持多种训练策略：支持NEFTune、packing等高级技术
4. 与transformers生态无缝集成：兼容HuggingFace全家桶
"""
from trl import (
    SFTTrainer,         # 监督微调训练器，封装了SFT训练的所有细节
    SFTConfig           # SFT训练配置类，继承自TrainingArguments并添加SFT特有参数
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

# 打印LoRA配置信息
print("LoRA配置:")
print(f"  秩(r): {peft_config.r}")
print(f"  目标模块: {peft_config.target_modules}")
print(f"  LoRA alpha: {peft_config.lora_alpha}")
print(f"  任务类型: {peft_config.task_type}")

# ====================== 4. 数据准备 ======================

"""
prompt-completion格式数据：
格式说明：
    - prompt: 包含用户输入和助手开始标记，格式为"<|im_start|>user\n用户问题<|im_end|>\n<|im_start|>assistant\n"
    - completion: 包含期望的助手回答和结束标记，格式为"助手回答<|im_end|>"
TRL的处理逻辑：
    1. 将prompt和completion拼接成完整序列：prompt + completion
    2. 自动计算损失掩码：prompt部分mask=0，completion部分mask=1
    3. 只对completion部分计算并反向传播梯度
优势：
    1. 明确分离：prompt部分（用户输入）不计算损失，completion部分（模型回答）计算损失
    2. TRL自动处理：SFTTrainer会自动识别并应用completion_only_loss
    3. 数据简洁：无需包含完整的对话历史，只需要当前对话的prompt和期望的completion
    4. 灵活性强：可以轻松处理多轮对话（将历史对话放在prompt中）
"""

# 简单的训练数据 - 使用prompt-completion格式,采用Qwen模型的ChatML格式
sample_data = [
    {
        "prompt": "<|im_start|>user\n什么是LoRA？<|im_end|>\n<|im_start|>assistant\n",
        "completion": "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数。<|im_end|>"
    },
    {
        "prompt": "<|im_start|>user\nLoRA有什么优势？<|im_end|>\n<|im_start|>assistant\n",
        "completion": "LoRA可以大幅减少训练参数，降低显存需求，同时保持较好的微调效果。<|im_end|>"
    },
    {
        "prompt": "<|im_start|>user\n什么是参数高效微调？<|im_end|>\n<|im_start|>assistant\n",
        "completion": "参数高效微调是指在微调大语言模型时，只更新一小部分参数，从而减少计算和存储需求的技术。<|im_end|>"
    },
    {
        "prompt": "<|im_start|>user\n如何使用LoRA微调模型？<|im_end|>\n<|im_start|>assistant\n",
        "completion": "使用LoRA微调模型需要配置LoRA参数（如秩r、目标模块等），然后将LoRA适配器添加到基础模型上进行训练。<|im_end|>"
    }
]

# 转换为Dataset格式
dataset = Dataset.from_list(sample_data)

# ====================== 5. 训练配置（使用TRL的SFTConfig） ======================

"""
SFTConfig高级参数：

1. completion_only_loss=True：
   - 功能：只计算completion部分的损失，自动屏蔽prompt部分
   - 实现：TRL内部会基于tokenizer的chat template或传入的格式自动计算损失掩码
   - 优势：无需手动实现复杂的损失掩码逻辑，代码更简洁

2. neftune_noise_alpha=10：
   - 功能：启用NEFTune（Noisy Embedding Fine-Tuning）噪声训练
   - 原理：在嵌入层添加均匀分布噪声，增强模型泛化能力
   - 效果：论文表明可提高对话模型的生成质量和多样性
   - 取值：通常在5-15之间，值越大噪声越强

3. dataset_text_field=None：
   - 功能：不使用单一的text字段，而是使用prompt和completion两个字段
   - 作用：告诉SFTTrainer数据是prompt-completion格式

4. packing=False：
   - 功能：不启用序列打包
   - 说明：packing=True可将多个短序列打包成长序列，提高训练效率
   - 权衡：可能影响收敛稳定性，小数据集建议关闭
"""
sft_config = SFTConfig(
    # 基础训练参数
    output_dir="./Qwen3_TRL_SFT_checkpoint",     # 模型和检查点的输出目录
    per_device_train_batch_size=1,    # 每个设备（如GPU）的训练批次大小
    gradient_accumulation_steps=4,    # 梯度累积步数：通过累积多个小批次实现大批次效果
                                      # 实际批次大小 = per_device_train_batch_size * gradient_accumulation_steps
    learning_rate=2e-4,               # 优化器的初始学习率（LoRA通常使用较大的学习率）
    num_train_epochs=2,               # 训练总轮数
    bf16=True,                        # 启用BF16混合精度训练（与torch_dtype=torch.bfloat16兼容）
    optim="adamw_bnb_8bit",           # 使用8-bit量化的AdamW优化器，大幅减少优化器显存占用
    lr_scheduler_type="cosine",       # 学习率调度器类型，使用余弦退火策略
    warmup_ratio=0.05,                # 学习率预热的比例，前5%的训练步数从0线性增加到初始学习率
    report_to="none",                 # 禁用外部日志报告（如W&B、TensorBoard），仅本地输出
    max_length=1024,                  # 最大序列长度，超过此长度的序列会被截断
    logging_steps=1,                  # 每1步打印一次日志
    
    # TRL特有高级参数
    neftune_noise_alpha=10,           # NEFTune噪声强度，10表示启用中等强度噪声
                                      # 工作原理：向嵌入向量添加Uniform(-alpha/sqrt(d), alpha/sqrt(d))噪声，其中d是嵌入向量的维度，alpha是噪声强度参数
    packing=False,                    # 是否启用序列打包（多个短序列合并成长序列）
    dataset_text_field=None,          # 不使用text字段，使用prompt和completion字段
    completion_only_loss=True,        # 显式设置为True以确保只计算assistant回复部分的损失
)

# ====================== 6. 训练模型（使用TRL的SFTTrainer） ======================

# 创建LoRA模型（在基础模型上应用LoRA配置）
peft_model = get_peft_model(base_model, peft_config)  # 冻结原始模型参数，只在指定模块插入可训练的LoRA参数
peft_model.print_trainable_parameters()  # 查看可训练参数统计信息

"""
SFTTrainer的prompt-completion处理机制：
1. 自动识别数据格式：当dataset_text_field=None且数据包含prompt和completion字段时
2. 内部处理流程：
   - 将prompt和completion拼接：full_text = prompt + completion
   - 使用tokenizer将full_text转换为input_ids
   - 计算prompt部分的长度，自动生成损失掩码
   - 只对completion部分计算损失
3. 无需手动准备labels或attention_mask
"""
trainer = SFTTrainer(
    model=peft_model,                  # 要训练的模型（LoRA模型）
    args=sft_config,                   # 训练配置参数
    train_dataset=dataset,             # 训练数据集（包含prompt和completion字段）
    processing_class=tokenizer         # 分词器，用于处理文本数据
)

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
    # 推理前配置：启用use_cache + 切换到eval模式
    trainer.model.eval()  # 切换到评估模式，禁用dropout等训练特有层
    trainer.model.config.use_cache = True  # 启用KV缓存加速推理

    # 构造符合Qwen格式的对话消息
    messages = [{"role": "user", "content": instruction}]
    
    # 应用聊天模板生成prompt
    prompt = tokenizer.apply_chat_template(
        messages,                    # 对话历史列表
        add_generation_prompt=True,  # 关键参数，在对话末尾添加"<|im_start|>assistant\n"标记，指示模型从这里开始生成回答
        tokenize=False               # 返回文本，不进行分词
        )
    # 分词并移动到模型所在设备
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
    
    # 生成回答（禁用梯度计算以节省显存）
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=100,        # 最大生成新token数
            temperature=0.7,           # 温度参数，控制随机性
            top_p=0.9,                 # 核采样参数，保留概率累积和达到0.9的token
            do_sample=True,            # 启用采样
            repetition_penalty=1.1     # 重复惩罚，>1.0降低重复概率
        )
    
    # 解码生成的完整文本
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)  # 保留特殊标记，便于提取assistant回答
    
    # 提取assistant的回答部分
    # Qwen格式：<|im_start|>assistant\n回答内容<|im_end|>
    assistant_start = "<|im_start|>assistant\n"
    assistant_end = "<|im_end|>"
    if assistant_start in full_text:
        # 提取assistant开始标记后的内容
        answer = full_text.split(assistant_start)[-1]
        # 去除结束标记
        if assistant_end in answer:
            answer = answer.split(assistant_end)[0]
    else:
        # 如果没有找到assistant标记，使用普通解码（跳过特殊标记）
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 恢复训练时的配置（以备后续可能的训练）
    trainer.model.config.use_cache = False
    
    return answer

# 测试推理 - 使用多个测试问题评估模型泛化能力
print("\n推理测试：")
test_questions = [
    "请用优美的话语夸赞和鼓励我：",  # 训练数据中类似的请求
    "请用夸张的话语夸赞我",         # 轻微变化
    "请用搞笑的话语夸赞我："        # 更多变化
]

for question in test_questions:
    print(f"我的询问：{question}")
    print(f"AI的回答：{generate_answer(question)}")
    print()

# ====================== 8.保存模型 ======================

save_path="./05_Qwen3_TRL_SFT_model"
# 保存完整的微调模型（包括LoRA适配器和基础模型配置）
# trainer.save_model()会自动保存LoRA权重和训练配置
trainer.save_model(save_path)
# 保存分词器配置
tokenizer.save_pretrained(save_path)
print(f"\n模型已保存到{save_path}")

"""
TRL库高级特性：
   a. completion_only_loss自动掩码：
      - 原理：基于prompt长度自动生成损失掩码
      - 实现：TRL内部计算tokenized序列中prompt部分的长度
      - 优势：无需手动实现复杂的损失掩码逻辑
      - 效果：精确控制模型只学习回答部分，避免学习用户输入
   b. NEFTune噪声训练：
      - 原理：向嵌入向量添加均匀分布噪声，增加训练多样性
      - 公式：e' = e + noise, noise ~ Uniform(-alpha/sqrt(d), alpha/sqrt(d))
      - 作用：类似于数据增强，提高模型泛化能力
      - 效果：论文表明可提升对话模型的生成质量和多样性
   c. prompt-completion数据格式：
      - 优势：数据准备简单，字段含义清晰
      - 处理：TRL自动拼接并计算损失掩码
      - 扩展：支持多轮对话（将历史对话放在prompt中）

训练流程优化：
   - 显存优化：4-bit量化 + LoRA + 8-bit优化器
   - 效率优化：梯度累积 + BF16混合精度
   - 效果优化：NEFTune + 余弦退火学习率

参数调优建议：
   a. LoRA参数：
        - r (秩)：增大r可增加模型容量，但也增加过拟合风险
        - lora_alpha：通常设为r的2倍，控制LoRA权重的影响
        - target_modules：可根据任务选择不同模块组合
   b. NEFTune参数：
        - neftune_noise_alpha：小数据集用10-15，大数据集用5-10
        - 可根据验证集表现调整，噪声过大可能影响收
   c. 训练参数：
        - learning_rate：LoRA通常用1e-4到3e-4
        - warmup_ratio：建议5-10%的步数用于预热
"""
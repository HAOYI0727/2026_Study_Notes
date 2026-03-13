"""
Name: Qwen3_simple_SFT.py
Description: 基于Qwen3-4B-Instruct模型的简单指令微调实现，采用手动参数冻结策略，仅训练最后2层的注意力与前馈网络参数，大幅降低显存占用。
"""

# ====================== 1. 镜像加速与导入必要库 ======================

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer 

# ====================== 2. 模型和分词器 ======================

# 清理显存和缓存，避免之前的计算占用资源
torch.cuda.empty_cache()
gc.collect()

# 指定模型名称，使用Qwen3-4B-Instruct版本
model_name = "Qwen/Qwen3-4B-Instruct-2507"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,  # 允许加载远程代码（Qwen使用了自定义模型结构）
    padding_side="right"     # 设置padding在右侧，因果语言模型通常需要右侧padding
)

# 设置pad_token（Qwen模型通常没有默认的pad_token，需要手动设置）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token           # 使用eos_token作为pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # 自动分配模型到可用设备（GPU/CPU）
    dtype=torch.bfloat16,    # 使用bfloat16精度，减少显存占用，同时保持数值稳定性
    trust_remote_code=True   # 允许加载自定义模型代码
).to("cuda")  # 显式移动到cuda

"""
# Qwen模型参数结构分析：
# 模型由embedding层、多个transformer层（layers）和输出层（lm_head）组成
# 每个transformer层包含：
#   - self_attn: 自注意力模块（q_proj, k_proj, v_proj, o_proj）
#   - mlp: 前馈网络模块（gate_proj, up_proj, down_proj）
#   - input_layernorm: 层归一化
#   - post_attention_layernorm: 注意力后层归一化

# 打印前10个参数名称，了解Qwen模型的参数结构
print("\nQwen模型参数名称结构示例：")
for i, (name, param) in enumerate(model.named_parameters()):
    print(f"{i}. {name}")
    if i >= 9:
        break

    Qwen模型参数名称结构示例：
    0. model.embed_tokens.weight          # 词嵌入层
    1. model.layers.0.self_attn.q_proj.weight    # 第0层注意力查询投影
    2. model.layers.0.self_attn.k_proj.weight    # 第0层注意力键投影
    3. model.layers.0.self_attn.v_proj.weight    # 第0层注意力值投影
    4. model.layers.0.self_attn.o_proj.weight    # 第0层注意力输出投影
    5. model.layers.0.self_attn.q_norm.weight    # 第0层查询归一化（Qwen特有）
    6. model.layers.0.self_attn.k_norm.weight    # 第0层键归一化（Qwen特有）
    7. model.layers.0.mlp.gate_proj.weight       # 第0层MLP门控投影（SwiGLU结构）
    8. model.layers.0.mlp.up_proj.weight         # 第0层MLP上升投影
    9. model.layers.0.mlp.down_proj.weight       # 第0层MLP下降投影
"""

# 参数冻结策略：仅训练最后几层的关键参数，大幅减少显存占用
# 这种策略基于迁移学习的思想：底层学习通用特征，高层学习任务特定特征
trainable_params = 0        # 可训练参数量
all_param = 0               # 总参数量
num_layers_to_train = 2     # 指定要训练的最后层数

# 获取模型总层数并计算起始训练层
total_layers = model.config.num_hidden_layers               # Qwen3-4B通常有36层
start_layer = max(0, total_layers - num_layers_to_train)    # 开始训练的层索引

# 遍历所有参数，设置requires_grad为False
for name, param in model.named_parameters():
    # 默认冻结所有参数
    param.requires_grad = False
    all_param += param.numel()  # 累加总参数数量
    
    # 1. 启用输出层（lm_head）训练，输出层直接映射到词表，对生成质量影响较大
    if "lm_head" in name:
        param.requires_grad = True
        trainable_params += param.numel()
        continue
    
    # 2. 启用最后num_layers_to_train层的关键参数训练
    if "model.layers." in name:
        # 从参数名中提取层号
        # 示例解析：model.layers.0.self_attn.q_proj.weight 
        # -> split("model.layers.")[1] = "0.self_attn.q_proj.weight"
        # -> split(".")[0] = "0"
        layer_part = name.split("model.layers.")[1].split(".")[0] 
        if layer_part.isdigit():  # 检查是否为数字字符串
            layer_num = int(layer_part)
            # 检查是否属于最后num_layers_to_train层
            if layer_num >= start_layer:
                # 检查是否是关键参数（注意力投影和MLP投影）
                # 选择这些参数的原因：注意力投影：负责信息提取和交互 + MLP投影：负责特征变换和非线性表达
                # 而不训练LayerNorm：参数较少，且对领域适应的贡献相对较小
                if any(key in name for key in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", 
                                               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]):
                    param.requires_grad = True
                    trainable_params += param.numel()

# 打印参数统计信息，监控训练参数比例
print(f"\ntrainable params: {trainable_params:,} ({trainable_params/all_param*100:.2f}%)")

# 使用AdamW优化器（权重衰减Adam），适合Transformer模型训练
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 使用较小的学习率防止灾难性遗忘

# ====================== 3. 训练数据 ======================

# 构建单轮对话训练数据，使用标准的ChatML格式，包含system、user、assistant三种角色
dialog = [
    {"role": "system", "content": "你是一个有用的助手"},  # 系统提示，定义助手角色
    {"role": "user", "content": "什么是LoRA？"},                     # 用户输入
    {"role": "assistant", "content": "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数。"}  # 期望输出
]

# 使用apply_chat_template方法将对话转换为模型可处理的输入格式，这是Hugging Face针对聊天模型的标准方法，自动添加特殊标记：
# Qwen3使用ChatML格式：
# <|im_start|>system\n...<|im_end|>\n
# <|im_start|>user\n...<|im_end|>\n
# <|im_start|>assistant\n...
input_ids = tokenizer.apply_chat_template(
                dialog,              # 对话历史列表
                return_tensors="pt"  # 返回PyTorch张量格式
            )

# 准备输入数据字典
input = {
    "input_ids": input_ids.to("cuda"),
    "attention_mask": torch.ones_like(input_ids).to("cuda")  # 全1张量，表示所有token都需要被关注（无padding）
}

# 设置labels，用于计算损失
# 在因果语言模型中，通常将labels设置为与input_ids相同，模型会自动屏蔽padding部分的损失
input["labels"] = input["input_ids"].clone()

# ====================== 4. 训练 ======================

# 前向传播计算损失
output = model(**input)
# 获取损失值并打印
loss = output.loss
print(f"Loss: {loss.item()}")

# 反向传播计算梯度
loss.backward()
# 优化器更新参数
optimizer.step()
# 清空梯度，准备下一步训练
optimizer.zero_grad()

# ====================== 5. 推理测试 ======================

print("\n推理测试：")
# 构建测试对话
test_dialog = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "用优美的词来夸一夸我吧"}
]

# 准备推理输入
test_input = tokenizer.apply_chat_template(
    test_dialog, 
    add_generation_prompt=True,  # 关键参数，推理时必须设置，在对话末尾添加生成提示标记，告诉模型从这里开始生成
                                 # 对于Qwen模型，这会添加"<|im_start|>assistant\n"，指示模型开始生成回答
    return_tensors="pt"
).to("cuda")

# 使用no_grad上下文管理器，禁用梯度计算以节省显存和加速推理
with torch.no_grad():
    test_output = model.generate(
        test_input,                                  # 输入token IDs
        max_new_tokens=500,                          # 最大生成新token数
        temperature=0.7,                             # 温度参数，控制随机性（>1更随机，<1更确定）
        attention_mask=torch.ones_like(test_input)   # attention mask，解决警告
    )

# 解码生成的回答部分
answer = tokenizer.decode(
            test_output[0][test_input.shape[-1]:],  # 切片获取新生成的token
            skip_special_tokens=True                # 跳过特殊标记（如<|im_end|>）
        )

print(f"user: {test_dialog[-1]['content']}")  # 打印用户问题
print(f"assistant: {answer}")                 # 打印模型回答

# 清理显存，释放资源
torch.cuda.empty_cache()
gc.collect()

# ====================== 6. 保存模型 ======================

# 保存微调后的模型和分词器
# 由于只训练了部分参数，实际上只有这些被更新的参数会保存，save_pretrained会保存完整的模型结构和参数（未训练的参数保持原样）
save_path="./01_Qwen3_simple_SFT_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n模型已保存到{save_path}")

"""
设计思路：
   - 采用参数冻结策略，仅训练最后2层的注意力投影和MLP投影
   - 基于迁移学习理论：底层学习通用语言特征，高层学习任务特定特征
   - 大幅减少可训练参数（约占总参数的2-5%），降低显存需求

使用方法：
   - 直接运行脚本进行单步训练（实际应用中应使用完整数据集和多轮训练）
   - 可根据需要调整num_layers_to_train参数控制训练层数
   - 可通过修改"any(key in name)"中的关键词选择不同参数组
"""
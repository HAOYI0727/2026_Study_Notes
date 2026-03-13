"""
Name: Qwen3_LoRA_SFT.py
Description: 基于Qwen3-4B-Instruct模型的LoRA指令微调实现，包含手动的损失掩码逻辑，确保只计算助手回答部分的损失，实现更精确的监督学习。
"""

# ====================== 1. 镜像加速与导入必要库 ======================

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

import torch
import gc
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import LoraConfig, TaskType, get_peft_model  # PEFT库用于实现LoRA

# ====================== 2. 模型和分词器 ======================

# 清理显存和缓存，避免之前的计算占用资源
torch.cuda.empty_cache()
gc.collect()

# 指定模型名称，使用Qwen3-4B-Instruct版本
model_name = "Qwen/Qwen3-4B-Instruct-2507"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,   # 允许加载远程代码（因为Qwen使用了自定义模型结构）
    padding_side="right"      # 设置padding在右侧，因果语言模型通常需要右侧padding
)

# 设置pad_token（Qwen模型通常没有默认的pad_token，需要手动设置）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token           # 使用eos_token作为pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # 自动分配模型到可用设备（GPU/CPU）
    dtype=torch.bfloat16,       # 使用bfloat16精度，减少显存占用，同时保持数值稳定性
    trust_remote_code=True      # 允许加载自定义模型代码
).to("cuda")  # 显式移动到cuda

# ====================== 3. 配置LoRA ======================

# LoRA (Low-Rank Adaptation) 原理：
# 通过在原始权重矩阵旁添加低秩可训练矩阵（A和B），实现参数高效微调
# 原始权重W保持不变，前向传播变为 h = Wx + BAx，其中B和A是可训练的低秩矩阵
peft_config = LoraConfig(
    r=8,                     # LoRA秩（rank），控制可训练参数数量，r越小参数越少
    target_modules=[         # 要应用LoRA的目标模块
        "self_attn.q_proj",  # 注意力查询投影
        "self_attn.k_proj",  # 注意力键投影
        "self_attn.v_proj",  # 注意力值投影
        "self_attn.o_proj",  # 注意力输出投影
        "mlp.gate_proj",     # MLP门控投影（SwiGLU结构）
        "mlp.up_proj",       # MLP上升投影
        "mlp.down_proj"      # MLP下降投影
    ],
    task_type=TaskType.CAUSAL_LM,   # 任务类型：因果语言模型
    lora_alpha=16,                  # LoRA缩放参数，控制低秩矩阵的权重，通常设置为r的2倍
    lora_dropout=0.05               # LoRA层的dropout比率，防止过拟合
)

# 将基础模型转换为PeftModel
model = get_peft_model(model, peft_config)   # 冻结原始模型参数，只将LoRA参数设置为可训练
model.print_trainable_parameters()           # 打印可训练参数统计信息

# 使用AdamW优化器（权重衰减Adam），适合Transformer模型训练
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# ====================== 4. 自定义Dataset和Collate函数 ======================

class SFTDataset:
    """
    简单的SFT数据集类——给原来的对话列表添加开头结尾的模板
    设计思路：将原始对话（字典列表）转换为带特殊标记的文本格式，为后续的分词和掩码计算做准备
    """
    def __init__(self, dialogs, tokenizer):
        """
        dialogs: 对话列表，每个对话是包含role和content的字典列表
        tokenizer: 分词器
        """
        self.dialogs = dialogs
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        """返回应用聊天模板后的文本"""
        dialog = self.dialogs[index]
        # 应用聊天模板但不添加生成提示
        chat_text = self.tokenizer.apply_chat_template(
            dialog, 
            tokenize=False,              # 返回文本而不是token IDs，便于后续处理
            add_generation_prompt=False  # 训练时不需要生成提示，只训练已有的回答部分
        )
        return chat_text
    
    def __len__(self):
        return len(self.dialogs)

"""
Chat模板应用示例说明：

原始输入:
dialog = [
    {"role": "user", "content": "用优美的词来夸一夸我吧"},
    {"role": "assistant", "content": "您如春日暖阳般温暖，如夏日清风般怡人"}
]
应用聊天模板后的输出对比:
| 参数设置 (add_generation_prompt) | 输出文本 (chat_text) | 用途说明 |
|-----|-----|-----|
| False | <|im_start|>user\n用优美的词来夸一夸我吧<|im_end|>\n<|im_start|>assistant\n您如春日暖阳般温暖，如夏日清风般怡人<|im_end|> | 用于训练：包含完整对话，没有后续生成提示 |
| True  | <|im_start|>user\n用优美的词来夸一夸我吧<|im_end|>\n<|im_start|>assistant\n您如春日暖阳般温暖，如夏日清风般怡人<|im_end|>\n<|im_start|>assistant | 用于推理：在完整对话后添加助手标记，提示模型开始生成 |

说明：
1. 当 add_generation_prompt=False 时，输出是完整的对话历史，用于训练模型学习已有的问答对
   当 add_generation_prompt=True 时，输出在完整对话历史末尾添加了 <|im_start|>assistant 标记，提示模型"现在该你回答了"，用于实际对话生成
2. 训练时使用False，希望模型学习整个对话（特别是assistant的回答）
   推理时使用True，需要模型生成新的回答
"""

def sft_collate(batch, tokenizer, max_length=500):
    """
    数据整理函数——创建损失掩码，只对助手回答部分计算损失
    核心处理：通过精确的损失掩码，确保模型只从助手回答部分学习，忽略system提示和用户问题部分，实现更精确的监督学习。
    
    参数说明：
    - batch: 文本列表，每个元素是一个完整的对话（包含system、user、assistant部分）
    - tokenizer: 分词器，用于将文本转换为模型可理解的数字
    - max_length: 最大序列长度，超过的部分会被截断
    返回值：
    - inputs: 包含tokenized后的输入数据（input_ids、attention_mask等）
    - loss_mask: 损失掩码，0表示不计算损失，1表示计算损失
    
    工作原理：首先将批量文本转换为数字序列，找到每个对话中助手开始回答的位置（通过搜索"<|im_start|>assistant"标记），
            然后创建掩码，只让助手回答部分的损失被计算，最后将数据转移到GPU上
    """
    
    # 1. 分词处理：将文本转换为数字序列，并进行padding和truncation
    inputs = tokenizer(
                batch, 
                max_length=max_length, 
                padding=True,           # 短文本会在末尾填充0，使所有样本长度一致
                truncation=True,        # 超过max_length的文本会被截断
                return_tensors="pt"     # 返回PyTorch张量格式
            )
    
    # 2. 识别assistant回答的起始位置
    # Qwen3模型使用特殊标记来区分不同角色的内容，助手开始的标记是：<|im_start|>assistant
    assistant_start_str = "<|im_start|>assistant"
    # 将标记转换为数字序列，add_special_tokens=False表示不添加额外的特殊标记
    assistant_start_ids = tokenizer(assistant_start_str, add_special_tokens=False)['input_ids']
    assistant_start_len = len(assistant_start_ids)
    
    # 3. 为每个样本创建损失掩码
    loss_mask = []  # 存储所有样本的损失掩码
    input_ids = inputs['input_ids']  # 获取tokenized后的输入数据
    
    # 遍历每个样本
    for i, input_id in enumerate(input_ids):
        # 初始化掩码：默认所有位置都不计算损失（用0表示） 
        mask = [0] * len(input_id)
        # 将输入数据转换为列表，方便后续操作
        input_id_list = input_id.tolist()
        
        # 查找assistant标记在当前样本中的位置
        # 从前往后搜索整个序列，使用滑动窗口匹配
        for j in range(len(input_id_list) - assistant_start_len + 1):
            # 检查当前位置是否匹配assistant标记
            if input_id_list[j : j+assistant_start_len] == assistant_start_ids:
                # 找到标记后，确定助手回答的实际起始位置
                assistant_content_start = j + assistant_start_len + 1  # Qwen模板格式：标记后面会有一个换行符（\n），所以需要+1跳过换行
                # 将助手回答部分的掩码设置为1，表示计算损失
                mask[assistant_content_start:] = [1] * (len(input_id_list) - assistant_content_start)  # 从回答起始位置到序列末尾都设置为1
                # 找到一个标记后就可以退出循环了（每个对话只有一个assistant部分）
                break
        
        # 将当前样本的掩码添加到列表中
        loss_mask.append(mask)
    
    # 4. 将数据转移到GPU上，加速后续计算
    # inputs是包含tokenized后数据的字典（如input_ids、attention_mask等）
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    # 将损失掩码也转移到GPU上
    loss_mask = torch.tensor(loss_mask, device="cuda")
    
    # 返回处理后的输入数据和损失掩码
    return inputs, loss_mask

# ====================== 5. 训练数据 ======================

# 示例训练数据：包含多个对话样本，每个样本都有完整的system-user-assistant结构
dialogs = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "什么是LoRA？"},
        {"role": "assistant", "content": "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数。"}
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "用优美的词来夸一夸我吧"},
        {"role": "assistant", "content": "您如春日暖阳般温暖，如夏日清风般怡人，如秋日明月般清朗，如冬日初雪般纯净。您的眼眸中蕴含着星辰大海，您的微笑里传递着无尽善意。"}
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Python有哪些优势？"},
        {"role": "assistant", "content": "Python具有简洁易读的语法、丰富的第三方库、跨平台兼容性、强大的社区支持、适合快速开发等优势。"}
    ]
]

# 创建数据集
dataset = SFTDataset(dialogs, tokenizer)

# 创建partial collate函数
# 使用functools.partial将tokenizer和max_length参数绑定到sft_collate函数，这样data_loader在调用collate_fn时就不需要再传递这些参数了
collate_fn = functools.partial(
    sft_collate,          # 要绑定的原始函数
    tokenizer=tokenizer,
    max_length=500 
)

# 创建数据加载器
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,           # 批次大小，由于只有少量示例数据，使用batch_size=1
    collate_fn=collate_fn,  # 使用已定义的partial函数进行数据整理
    shuffle=True            # 每个epoch打乱数据顺序，提高训练效果
)

# ====================== 6. 训练 ======================

epochs = 3  # 训练轮数

for epoch in range(epochs):
    for step, (inputs, loss_mask) in enumerate(data_loader):
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits  # 形状: [batch_size, sequence_length, vocab_size]
        
        # 计算损失，只对回答部分计算
        # 关键步骤：需要将logits和labels对齐，因为模型预测的是下一个token
        
        # 将logits向右偏移一位（移除最后一个token），模型对位置i的logits预测的是位置i+1的token
        shift_logits = logits[:, :-1, :].contiguous()  # 形状: [batch_size, sequence_length-1, vocab_size]
        # 将labels向左偏移一位（移除第一个token），因果语言模型使用输入作为目标
        shift_labels = inputs["input_ids"][:, 1:].contiguous()  # 形状: [batch_size, sequence_length-1]
        # 将损失掩码向左偏移一位（与labels对齐），确保掩码与偏移后的labels一一对应
        shift_loss_mask = loss_mask[:, 1:].contiguous()  # 形状: [batch_size, sequence_length-1]
        
        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(
                        ignore_index=tokenizer.pad_token_id,    # 忽略padding位置的损失
                        reduction="none"                        # 返回每个位置的损失，不自动聚合
                    )
        # 将logits和labels展平为二维和一维，以便计算交叉熵
        loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),   # 形状: [batch_size*(sequence_length-1), vocab_size]
                    shift_labels.view(-1)                           # 形状: [batch_size*(sequence_length-1)]
                )  # 形状: [batch_size*(sequence_length-1)]
        # 应用损失掩码：只保留助手回答部分的损失
        loss = loss * shift_loss_mask.view(-1)  # 形状不变，非回答部分被置0
        # 计算平均损失：损失总和除以掩码中1的数量
        loss = loss.sum() / (shift_loss_mask.sum() + 1e-8)  # 添加1e-8防止除以0，形状为标量
        print(f"Epoch: {epoch+1}, Step: {step+1}, Loss: {loss.item()}")
        
        # 反向传播和优化
        loss.backward()          # 计算梯度
        optimizer.step()         # 更新参数
        optimizer.zero_grad()    # 清空梯度，准备下一步训练

# ====================== 7. 推理测试 ======================

print("\n推理测试：")

# 构建测试对话，使用与训练不同的用户输入来评估模型泛化能力
test_dialog = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用最优美的话语夸赞和鼓励我："}
]

# 准备推理输入
test_input = tokenizer.apply_chat_template(
    test_dialog, 
    add_generation_prompt=True,  # 推理时必须设置，对话末尾添加"<|im_start|>assistant\n"，指示模型开始生成
    return_tensors="pt"
).to("cuda")

# 使用no_grad上下文管理器，禁用梯度计算以节省显存和加速推理
with torch.no_grad():
    test_output = model.generate(
        test_input,                                  # 输入token IDs
        max_new_tokens=200,                          # 最大生成新token数
        temperature=0.7,                             # 温度参数，控制随机性（>1更随机，<1更确定）
        attention_mask=torch.ones_like(test_input)   # attention mask，确保正确生成
    )

# 解码生成的回答部分
answer = tokenizer.decode(
            test_output[0][test_input.shape[-1]:],   # 切片获取新生成的token
            skip_special_tokens=True                 # 跳过特殊标记（如<|im_end|>）
        )

print(f"user: {test_dialog[-1]['content']}")  # 打印用户问题
print(f"assistant: {answer}")                 # 打印模型回答

# 清理显存，释放资源
torch.cuda.empty_cache()
gc.collect()

# ====================== 8. 保存模型 ======================

# 保存微调后的LoRA模型
# LoRA只保存可训练的低秩矩阵，文件大小很小（通常几MB）
save_path="./02_Qwen3_LoRA_SFT_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nLoRA模型已保存到{save_path}")

"""
设计思想：
   - 采用LoRA参数高效微调，大幅减少可训练参数（通常<1%）
   - 实现精确的损失掩码，确保模型只从助手回答部分学习
   - 保持system提示和用户问题不变，专注于优化回答质量

实现重点：
   - 损失掩码机制：通过识别"<|im_start|>assistant"标记，精确定位回答区域
   - LoRA适配：在注意力层和MLP层添加低秩矩阵，实现高效微调
   - 数据流设计：使用functools.partial优雅地处理collate函数参数绑定

使用方法：
   - 准备对话数据：每个样本包含system、user、assistant三部分
   - 调整超参数：可修改LoRA的r、lora_alpha等参数控制微调强度
   - 多轮训练：代码已实现多epoch训练，可根据需要调整epochs
   - 模型保存与加载：保存的LoRA权重可以加载到基础模型上使用
"""
import json
import os
import sys
import time
import torch.onnx
import pandas as pd
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from entmax import entmax_bisect
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from transformers import LongformerModel, LongformerConfig, AutoTokenizer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf8', buffering=1)




class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_window, use_per_head_params=False):
        super(FlashAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.attention_window = attention_window
        self.use_per_head_params = use_per_head_params

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, alpha=None, span=None):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x).reshape(
            batch_size, seq_length, self.num_heads, 3 * self.head_dim
        ).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_output = self.flash_attention(q, k, v, alpha=alpha, span=span)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)
        attn_output = self.layer_norm(attn_output)
        return self.o_proj(attn_output)

    def flash_attention(self, q, k, v, alpha=1.5, span=None):
        batch_size, num_heads, seq_length, head_dim = q.size()
        attn_output = torch.zeros_like(q)

        # 处理 alpha 参数
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.view(1, -1, 1, 1).to(q.device)  # [1, num_heads, 1, 1]

        for i in range(0, seq_length, self.attention_window):
            q_start = i
            q_end = min(i + self.attention_window, seq_length)

            # 计算键/值块范围
            if span is not None:
                if isinstance(span, torch.Tensor):
                    # 训练时保持张量，推理时转为标量
                    if not self.training:
                        span_val = int(span.mean().item())
                    else:
                        span_val = span.mean()
                    k_start = max(0, q_start - int(span_val))
                    k_end = min(seq_length, q_end + int(span_val))
                else:
                    k_start = max(0, q_start - span)
                    k_end = min(seq_length, q_end + span)
            else:
                k_start = q_start
                k_end = q_end

            q_chunk = q[:, :, q_start:q_end, :]
            k_chunk = k[:, :, k_start:k_end, :]
            v_chunk = v[:, :, k_start:k_end, :]

            # 计算注意力分数
            attn_scores = torch.einsum('bhqd,bhkd->bhqk', q_chunk, k_chunk) / (head_dim ** 0.5)

            # 应用跨度掩码
            if span is not None:
                rel_pos = torch.arange(q_chunk.size(2), device=q.device)[:, None] - \
                          torch.arange(k_chunk.size(2), device=q.device)[None, :]
                if isinstance(span, torch.Tensor):
                    # 调整 span 形状为 [1, num_heads, 1, 1]
                    span_ = span.view(1, -1, 1, 1).to(q.device)
                    # 扩展相对位置矩阵到 [1, 1, Q, K]
                    rel_pos_expanded = rel_pos.abs().unsqueeze(0).unsqueeze(0)  # [1, 1, Q, K]
                    # 广播比较生成掩码 [1, num_heads, Q, K]
                    pos_mask = rel_pos_expanded > span_
                    # 调整掩码形状为 [batch, heads, Q, K]
                    pos_mask = pos_mask.expand(batch_size, -1, -1, -1)
                else:
                    pos_mask = rel_pos.abs() > span
                attn_scores = attn_scores.masked_fill(pos_mask, float('-inf'))

            # 计算注意力权重
            attn_weights = entmax_bisect(attn_scores, alpha=alpha, dim=-1)
            chunk_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v_chunk)
            attn_output[:, :, q_start:q_end, :] = chunk_output

        return attn_output

class MixAdaptiveAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_window=1024):
        super(MixAdaptiveAttention, self).__init__()
        self.flash_attention_layer = FlashAttention(
            embed_dim, num_heads, attention_window, use_per_head_params=True
        )
        self.alpha = nn.Parameter(torch.ones(num_heads) * 1.5 + torch.randn(num_heads) * 0.05)
        self.attn_span = nn.Parameter(torch.ones(num_heads) * 100.0 + torch.randn(num_heads) * 1.0)

    def forward(self, x):
        attn_output = self.flash_attention_layer(
            x,
            alpha=self.alpha,
            span=self.attn_span
        )
        return attn_output

    def clamp_parameters(self):
        with torch.no_grad():
            self.alpha.clamp_(1.0, 2.0)
            self.attn_span.clamp_(min=1.0)


class CustomLongformer(nn.Module):
    def __init__(self, config):
        super(CustomLongformer, self).__init__()
        self.longformer = LongformerModel(config)
        self.attention_heads = nn.ModuleList([
            MixAdaptiveAttention(config.hidden_size, config.num_attention_heads)
            for _ in range(config.num_hidden_layers)
        ])
        self.adjustment_layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.adjustment_layer2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.config = config

    def forward(self, input_ids, attention_mask):
        if self.training:
            for layer in self.attention_heads:
                layer.clamp_parameters()
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        rev_output = self.adjustment_layer1(hidden_states)
        for attn_head in self.attention_heads:
            rev_output = attn_head(rev_output)
        rev_output = self.adjustment_layer2(rev_output)
        logits = self.linear(rev_output)
        return logits


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024, file_type=".txt"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_type = file_type

        if self.file_type == ".txt":
            self.data = self._load_txt(file_path)
        elif self.file_type == ".parquet":
            self.data = self._load_parquet(file_path)
        elif self.file_type == ".csv":
            self.data = self._load_csv(file_path)
        elif self.file_type == ".json":
            self.data = self._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if len(self.data) == 0:
            raise ValueError("Loaded dataset is empty.")

    def _load_txt(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.readlines()
        except Exception as e:
            raise RuntimeError(f"Failed to load TXT file: {str(e)}")

    def _load_parquet(self, file_path):
        df = pd.read_parquet(file_path)
        if "text" in df.columns:
            return df["text"].tolist()
        raise ValueError("Parquet file must contain a 'text' column.")

    def _load_csv(self, file_path):
        df = pd.read_csv(file_path)
        if "text" in df.columns:
            return df["text"].tolist()
        raise ValueError("CSV file must contain a 'text' column.")

    def _load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            squad_data = json.load(f)

        records = []
        for article in squad_data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    answers = [answer["text"] for answer in qa["answers"]]
                    records.append({"context": context, "question": question, "answers": answers})
        return records

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        # 检查是否有 context 和 question 字段
        if "context" not in record or "question" not in record:
            print(f"Record at index {idx} is invalid: {record}")
            return None

        context = record["context"]
        question = record["question"]
        text = context + " " + question  # 拼接问题和上下文

        # 使用 tokenizer 进行编码
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()

        # 设置一个占位符标签，替换为你的真实标签逻辑
        label = input_ids.clone()
        label[:-1] = input_ids[1:]
        label[-1] = -100  # 忽略最后一个位置
        return input_ids, attention_mask, label

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.default_collate(batch)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)
        init.zeros_(m.bias)


def log_gpu_info(start_time, log_file):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    end_time = time.time()
    elapsed_time = end_time - start_time
    sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)

    # 获取利用率
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu

    # 获取温度
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

    # 获取 SM 数量：改用 nvmlDeviceGetAttribute 获取多处理器数量
    sm_count = 24
    cores_per_sm=128
    sm_clock_ghz = sm_clock / 1000
    # 理论FP32算力估算（单位：GFLOPS），公式：SM数量 × 每个SM的核心数 × (SM时钟频率 [GHz]) × 2
    flops = sm_count * cores_per_sm * sm_clock_ghz * 2
    flops=flops/1000
    with open(log_file, "a") as f:
        f.write(f"{elapsed_time:.4f}\t{mem_info.used // 1024 ** 2}MB\t{flops}\t{temp}\t{gpu_util}")
    pynvml.nvmlShutdown()


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1).expand(-1, outputs.size(1))

            loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
            total_loss += loss.item()  # 累加到 total_loss
        val_loss = total_loss / len(dataloader)  # 平均损失
    return val_loss


def train(model, dataloader, optimizer, alpha_optimizers, span_optimizers, scheduler, alpha_schedulers, span_schedulers,
          epoch, device, name, lmbda=0.0001, grad_clip=1, start_time=0, scaler=None, save_checkpoint_interval=5):
    model.train()
    log_dir = "C:\\Users\\27369\\PycharmProjects\\graduate\\log\\"
    log_dir = os.path.join(log_dir, f"{name}\\")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 确保日志目录存在

    # 动态生成日志文件名
    log_file = os.path.join(log_dir, f"{name}_{epoch}_train_log.txt")
    alpha_span_log_file = os.path.join(log_dir, f"{name}_{epoch}_alpha_span_log.txt")

    temp_directory = 'C:\\Users\\27369\\PycharmProjects\\graduate\\model\\temp\\'
    temp_directory= os.path.join(temp_directory, f"{epoch}\\")
    if not os.path.exists(temp_directory):  # 检查目录是否存在
        os.makedirs(temp_directory)
    checkpoint_path = os.path.join(temp_directory, f"{name}.pt")
    train_loss = 0

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        for alpha_optimizer in alpha_optimizers:
            alpha_optimizer.zero_grad()
        for span_optimizer in span_optimizers:
            span_optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1).expand(-1, outputs.size(1))

            loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

            # 计算 L2 正则化项
            l2_regularization = torch.tensor(0., device=device)
            for name, param in model.named_parameters():
                if 'alpha' in name or 'attn_span' in name:
                    l2_regularization += torch.norm(param) ** 2

            # 计算总损失
            total_loss = loss + lmbda * l2_regularization
            train_loss += total_loss.item()

        # 使用自动混合精度进行梯度计算
        scaler.scale(total_loss).backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # 梯度更新
        scaler.step(optimizer)
        scaler.update()

        # 更新 alpha 和 span 参数的优化器
        for alpha_optimizer in alpha_optimizers:
            scaler.step(alpha_optimizer)
            scaler.update()
        for span_optimizer in span_optimizers:
            scaler.step(span_optimizer)
            scaler.update()

        # 更新学习率调度器
        scheduler.step(loss)
        for alpha_scheduler in alpha_schedulers:
            alpha_scheduler.step(loss)
        for span_scheduler in span_schedulers:
            span_scheduler.step(loss)

        # 记录日志

        if batch_idx % 10 == 0:
            total_elapsed_time = time.time() - start_time
            with open(log_file, "a") as f:
                if batch_idx == 0:
                    header = [
                        "Train Epoch\tBatch\tBatch(%)\tLoss\tTime(s)\tOptimizer LR\tAlpha Optimizer LR\tSpan Optimizer LR\tBatch Processing\telapsed_time\tmem_info.used\tflops\tTemp\tGPU_Util(%)\tValidation Epoch\tLoss"]
                    f.write("\t".join(header) + "\n")
                f.write(
                    f"{epoch}\t{batch_idx * len(input_ids)}\t{100. * batch_idx / len(dataloader):.8f}\t{loss.item():.8f}\t{total_elapsed_time:.4f}\t{optimizer.param_groups[0]['lr']}\t{alpha_optimizer.param_groups[0]['lr']}\t{span_optimizer.param_groups[0]['lr']}\t")
            log_gpu_info(start_time, log_file)
            with open(alpha_span_log_file, "a") as f:
                # 如果是第一个 Batch，先写入标题行
                if batch_idx == 0:
                    header = ["epoch\tBatch"]
                    for param_name, param in model.named_parameters():
                        if 'alpha' in param_name or 'attn_span' in param_name:
                            header.append(f"{param_name}_param")
                            header.append(f"{param_name}_grad")
                    f.write("\t".join(header) + "\n")

                # 写入当前 Batch 的数据行
                row = [f"{epoch}\t{batch_idx * len(input_ids)}"]
                for param_name, param in model.named_parameters():
                    if 'alpha' in param_name or 'attn_span' in param_name:
                        # 获取参数值和梯度值（梯度为 None 时用空值代替）
                        param_value = param.detach().cpu().numpy().flatten()  # 将多维数组展开为一维
                        gradient_value = (
                            param.grad.detach().cpu().numpy().flatten()
                            if param.grad is not None else None
                        )
                        # 拼接参数和梯度值，以空格分隔多维数值
                        row.append("\t".join(map(str, param_value)))
                        row.append("\t".join(map(str, gradient_value)) if gradient_value is not None else "")
                f.write("\t".join(row) + "\n")
            # 验证模型
            val_loss = validate(model, dataloader, device)
            with open(log_file, "a") as f:
                f.write(f"\t{epoch}\t{val_loss:.8f}\n")

        # 使用验证损失调整学习率
        scheduler.step(val_loss)
        for alpha_scheduler in alpha_schedulers:
            alpha_scheduler.step(val_loss)
        for span_scheduler in span_schedulers:
            span_scheduler.step(val_loss)

        if epoch % save_checkpoint_interval == 0 and epoch != 0:
            torch.save(model.state_dict(), checkpoint_path)

    return train_loss


def choose_dataset_loading_method():
    print("请选择数据集加载方式:")
    print("1. 从 Hugging Face 项目下载")
    print("2. 从本地文件加载")
    choice = input("请输入选择: ")

    if choice == '1':
        train_dataset_name = input("请输入训练集名称: ")
        train_split = input("请输入训练集拆分（如 'train'）: ")
        val_dataset_name = input("请输入测试集名称: ")
        val_split = input("请输入测试集拆分（如 'val'）: ")
        return "huggingface", (train_dataset_name, train_split), (val_dataset_name, val_split)
    elif choice == '2':
        train_file_path = input("请输入训练集文件路径: ")
        val_file_path = input("请输入测试集文件路径: ")

        # 根据文件类型加载数据集
        train_file_type = os.path.splitext(train_file_path)[-1].lower()
        val_file_type = os.path.splitext(val_file_path)[-1].lower()

        return "local_file", (train_file_path, train_file_type), (val_file_path, val_file_type)
    else:
        print("无效选择，请重新选择")
        return choose_dataset_loading_method()



def find_model_files(base_path):
    pt_files = []
    onnx_files = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.join(root, file))
            elif file.endswith(".onnx"):
                onnx_files.append(os.path.join(root, file))

    return pt_files, onnx_files


def print_model_files():
    base_path = "C:\\Users\\27369\\PycharmProjects\\graduate\\model\\"
    pt_files, onnx_files = find_model_files(base_path)

    print("Found .pt files:")
    for file in pt_files:
        print(file)

    print("\nFound .onnx files:")
    for file in onnx_files:
        print(file)


def main():
    start_time = time.time()
    scaler = GradScaler()

    print_model_files()

    choice = input("请输入选择: 键入1：训练已有模型，然后输入模型路径；键入2：训练一个新模型\n")
    if choice == '1':
        model_path = input("请输入已有模型的路径: ")
        if not os.path.exists(model_path):
            print(f"模型路径 {model_path} 不存在，程序将退出。")
            return
        try:
            print(f"Loading model from {model_path}")
            model = torch.load(model_path)
        except Exception as e:
            print(f"加载模型时出现错误: {e}")
            return
    elif choice == '2':
        print("Initializing a new model")
        tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\27369\\PycharmProjects\\graduate\\tokennizer\\")
        config = LongformerConfig.from_pretrained("C:\\Users\\27369\\PycharmProjects\\graduate\\tokennizer\\",
                                                  vocab_size=tokenizer.vocab_size)
        config.num_hidden_layers = 32
        config.hidden_size = 768
        config.num_attention_heads = 8
        config.attention_window = [1024] * config.num_hidden_layers
        model = CustomLongformer(config)
        model.apply(initialize_weights)
    else:
        print("无效输入，程序将退出。")
        return

    # 选择数据集加载方式
    loading_method, train_args, val_args = choose_dataset_loading_method()

    tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\27369\\PycharmProjects\\graduate\\tokennizer\\")

    if loading_method == "huggingface":
        train_dataset_name, train_split = train_args
        val_dataset_name, val_split = val_args
        try:
            train_dataset = TextDataset(train_dataset_name, train_split, tokenizer)
            val_dataset = TextDataset(val_dataset_name, val_split, tokenizer)
        except Exception as e:
            print(f"加载 Hugging Face 数据集时出现错误: {e}")
            return
    elif loading_method == "local_file":
        train_file_path, train_file_type = train_args
        val_file_path, val_file_type = val_args
        try:
            train_dataset = TextDataset(train_file_path, tokenizer, file_type=train_file_type)
            val_dataset = TextDataset(val_file_path, tokenizer, file_type=val_file_type)
        except Exception as e:
            print(f"加载本地数据集时出现错误: {e}")
            return
    else:
        print("数据集加载方式无效")
        return

    assert train_dataset is not None and val_dataset is not None, "未成功加载数据集"

    name = input("请输入模型名称：")

    data_len = 0.001
    train_size = int(data_len * len(train_dataset))
    val_size = int(data_len * len(val_dataset))

    train_indices = list(range(train_size))
    val_indices = list(range(val_size))

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    batch_size = 1
    seq_length = 1024

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length)).to(device)
    dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)

    summary(model, input_data=(dummy_input_ids, dummy_attention_mask), depth=16,
            col_names=["input_size", "output_size", "num_params", "trainable"], dtypes=[torch.long, torch.long])

    optimizer = AdamW([p for n, p in model.named_parameters() if 'alpha' not in n and 'attn_span' not in n], lr=0.0001)
    alpha_optimizers = [AdamW([p for n, p in attn_head.named_parameters() if 'alpha' in n], lr=0.1) for attn_head in
                        model.attention_heads]
    span_optimizers = [AdamW([p for n, p in attn_head.named_parameters() if 'attn_span' in n], lr=0.1) for attn_head
                       in
                       model.attention_heads]

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    alpha_schedulers = [ReduceLROnPlateau(alpha_optimizer, mode='min', factor=0.1, patience=2, verbose=True) for
                        alpha_optimizer in alpha_optimizers]
    span_schedulers = [ReduceLROnPlateau(span_optimizer, mode='min', factor=0.1, patience=2, verbose=True) for
                       span_optimizer in span_optimizers]

    num_epochs = 10

    writer = SummaryWriter()  # 初始化 TensorBoard
    for epoch in range(num_epochs):
        # 训练阶段
        train_loss = train(model, train_loader, optimizer, alpha_optimizers, span_optimizers,
                           scheduler, alpha_schedulers, span_schedulers, epoch,
                           device, name, start_time=start_time, scaler=scaler)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # 验证阶段
        val_loss = validate(model, val_loader, device)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # 更新学习率
        scheduler.step(val_loss)
        for alpha_scheduler in alpha_schedulers:
            alpha_scheduler.step(val_loss)
        for span_scheduler in span_schedulers:
            span_scheduler.step(val_loss)

    writer.close()  # 关闭 TensorBoard

    pt_directory = 'C:\\Users\\27369\\PycharmProjects\\graduate\\model\\.pt\\'
    if not os.path.exists(pt_directory):  # 检查目录是否存在
        os.makedirs(pt_directory)  # 如果不存在则创建

    model_path = os.path.join(pt_directory, f"{name}.pt")

    # 保存模型
    try:
        torch.save(model, model_path)
        print(f"模型已成功保存到：{model_path}")
    except Exception as e:
        print(f"保存模型时出错：{e}")

    param_count = sum(p.numel() for p in model.parameters())
    total_time = time.time() - start_time
    print(f"Total number of parameters: {param_count}")
    print(f"Total training time: {total_time:.4f}s")
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Number of hidden layers: {model.config.num_hidden_layers}")
    print(f"Vocabulary size: {model.config.vocab_size}")
    print(f"Attention window size: {model.config.attention_window}")
if __name__ == "__main__":
    main()

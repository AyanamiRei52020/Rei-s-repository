import json
import os
import sys
import time
import datetime
import torch.onnx
import pandas as pd
import pynvml
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from entmax import entmax_bisect
from adaptive_span import AdaptiveSpan
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import TransfoXLModel,TransfoXLConfig, AutoTokenizer
os.environ["TRUST_REMOTE_CODE"] = "True"
checkpoint = 'transfo-xl/transfo-xl-wt103'
revision = '40a186da79458c9f9de846edfaea79c412137f97'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf8', buffering=1)



class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_window):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.attention_window = attention_window

        self.qkv_proj   = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj     = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, alpha=None, span: AdaptiveSpan=None):
        """
        x: [B, L, D]
        alpha: float or Tensor([num_heads])
        span: AdaptiveSpan 实例
        """
        B, L, D = x.size()  # 在此定义 D，供最后 reshape 使用

        # 1) QKV 投影并拆分
        qkv = (self.qkv_proj(x)
               .view(B, L, self.num_heads, 3 * self.head_dim)
               .permute(0, 2, 1, 3))     # [B, H, L, 3*head_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # 各自 [B, H, L, head_dim]

        # 2) Optional: 用 AdaptiveSpan 裁剪缓存
        if span is not None:
            k, v, _ = span.trim_memory(q, k, v, None)

        # 3) Windowed Attention
        out = torch.zeros_like(q)        # [B, H, L, head_dim]
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.view(1, -1, 1, 1).to(q.device)

        for start in range(0, L, self.attention_window):
            end = min(start + self.attention_window, L)
            q_chunk = q[:, :, start:end, :]   # [B, H, W, head_dim]
            # 这里直接用全局 k/v 的对应区间
            k_chunk = k[:, :, max(0, start):min(k.size(2), end), :]
            v_chunk = v[:, :, max(0, start):min(v.size(2), end), :]

            # 3.1) 计算原始 scores
            scores = torch.einsum('bhqd,bhkd->bhqk', q_chunk, k_chunk) \
                     / (self.head_dim ** 0.5)  # [B, H, W, W']

            # 3.2) 展平后调用 AdaptiveSpan 掩码 + 归一化
            if span is not None:
                B_, H_, Q_, K_ = scores.shape
                # 展平到 [B*H, Q, K]
                scores_flat = scores.reshape(B_ * H_, Q_, K_)
                # AdaptiveSpan.forward 接受三维张量 :contentReference[oaicite:2]{index=2}
                scores_masked = span(scores_flat, normalize=True)
                # 恢复到 [B, H, Q, K]
                scores = scores_masked.reshape(B_, H_, Q_, K_)

            # 3.3) entmax + 加权求和
            weights   = entmax_bisect(scores, alpha=alpha, dim=-1)
            chunk_out = torch.einsum('bhqk,bhkd->bhqd', weights, v_chunk)
            out[:, :, start:end, :] = chunk_out

        # 4) 恢复到 [B, L, D] 并加 LayerNorm + 输出投影
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        out = self.layer_norm(out)
        return self.o_proj(out)

class MixAdaptiveAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_window, adaptive_span_config):
        super().__init__()
        self.flash = FlashAttention(embed_dim, num_heads, attention_window)
        self.alpha = nn.Parameter(torch.ones(num_heads)*1.5 +
                                  torch.randn(num_heads)*0.05)
        self.adaptive_span = AdaptiveSpan(
            attn_span        = adaptive_span_config['attn_span'],
            adapt_span_loss  = adaptive_span_config['adapt_span_loss'],
            adapt_span_ramp  = adaptive_span_config['adapt_span_ramp'],
            adapt_span_init  = adaptive_span_config.get('adapt_span_init', 0.0),
            adapt_span_cache = adaptive_span_config.get('adapt_span_cache', True),
            nb_heads         = num_heads
        )

    def forward(self,
                hidden_states,            # [batch, seq_len, dim]
                attention_mask=None,      # HF 会传进来
                output_attentions=False,  # HF 会传进来
                **kwargs):                # 捕获其他可能的参数
        # 训练时 clamp span
        self.adaptive_span.clamp_param()

        # 调用自定义 FlashAttention 只需要 hidden_states
        attn_output = self.flash(hidden_states,
                                 alpha=self.alpha,
                                 span=self.adaptive_span)

        # 构造返回值：tuple，第0项是 context，后面可按 output_attentions 加入 attn_probs
        outputs = (attn_output, )
        if output_attentions:
            # 目前没有返回 attn_weights，返回 None 占位
            outputs += (None, )

        return outputs



class CustomXLformer(nn.Module):
    def __init__(self, config, adaptive_span_config):
        super().__init__()
        self.TransfoXLModel = TransfoXLModel(config)
        self.config = config

        # 按层替换 self-attention，并取出对应的 attention_window
        for idx, layer in enumerate(self.TransfoXLModel.encoder.layer):
            # 如果 attention_window 是 list，就用第 idx 个；否则当作单个整数
            if isinstance(config.attention_window, (list, tuple)):
                window_size = config.attention_window[idx]
            else:
                window_size = config.attention_window

            layer.attention.self = MixAdaptiveAttention(
                embed_dim            = config.hidden_size,
                num_heads            = config.num_attention_heads,
                attention_window     = window_size,
                adaptive_span_config = adaptive_span_config
            )

        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask):
        if self.training:
            # clamp span 参数
            for layer in self.lTransfoXLModel.encoder.layer:
                layer.attention.self.adaptive_span.clamp_param()

        outputs = self.TransfoXLModel(input_ids, attention_mask=attention_mask)
        logits   = self.linear(outputs.last_hidden_state)
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
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    # batch: List[ (input_ids, attention_mask, label) ]
    input_ids, attention_masks, labels = zip(*batch)
    # 每个都是 tuple of tensors，转成 [B, ...] 的大 tensor
    input_ids      = torch.stack(input_ids, dim=0)
    attention_masks= torch.stack(attention_masks, dim=0)
    labels         = torch.stack(labels, dim=0)
    return input_ids, attention_masks, labels


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
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1).expand(-1, outputs.size(1))

            loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1),ignore_index=-100)
            total_loss += loss.item()  # 累加到 total_loss
        val_loss = total_loss / len(dataloader)  # 平均损失
    return val_loss


def train(model, dataloader, optimizer, alpha_optimizers, span_optimizers, scheduler, alpha_schedulers, span_schedulers,
          epoch, device, name, lmbda=0.00001, grad_clip=1, start_time=0, scaler=None, save_checkpoint_interval=5):
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

    total_loss = 0

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
                if 'alpha' in name or 'adaptive_span' in name:
                    l2_regularization += torch.norm(param) ** 2

            # 计算总损失
            total_loss = loss + lmbda * l2_regularization

        # 使用自动混合精度进行梯度计算
        scaler.scale(total_loss).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
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


        # 记录日志

        if batch_idx %1==0 :
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
                        if 'alpha' in param_name or 'span' in param_name:
                            header.append(f"{param_name}_param")
                            header.append(f"{param_name}_grad")
                    f.write("\t".join(header) + "\n")

                # 写入当前 Batch 的数据行
                row = [f"{epoch}\t{batch_idx * len(input_ids)}"]
                for param_name, param in model.named_parameters():
                    if 'alpha' in param_name or 'span' in param_name:
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

        val_loss = validate(model, dataloader, device)
        scheduler.step(val_loss)
        for alpha_scheduler in alpha_schedulers:
            alpha_scheduler.step(val_loss)
        for span_scheduler in span_schedulers:
            span_scheduler.step(val_loss)
        with open(log_file, "a") as f:
            f.write(f"\t{epoch}\t{val_loss:.8f}\n")
    if epoch % save_checkpoint_interval == 0 and epoch != 0:
        torch.save(model.state_dict(), checkpoint_path)
    train_loss=total_loss/len(dataloader)

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
        tokenizer = AutoTokenizer.from_pretrained("transfo-xl/transfo-xl-wt103")
        config = TransfoXLConfig(vocab_size=tokenizer.vocab_size)
        config.num_hidden_layers = 32
        config.hidden_size = 768
        config.num_attention_heads = 8
        config.attention_window = [512] * config.num_hidden_layers
        adaptive_span_config = {
            'attn_span': config.max_position_embeddings,
            'adapt_span_ramp': 128,
            'adapt_span_loss': 0.1
        }
        model = TransfoXLModel(config,adaptive_span_config)

        model.apply(initialize_weights)
    else:
        print("无效输入，程序将退出。")
        return

    # 选择数据集加载方式
    loading_method, train_args, val_args = choose_dataset_loading_method()

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

    name = input("请输入模型名称：") + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_len = 0.0001
    train_size = int(data_len * len(train_dataset))
    val_size = int(data_len * len(val_dataset))

    train_indices = list(range(train_size))
    val_indices = list(range(val_size))

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    batch_size = 1

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)





    optimizer = AdamW([p for n, p in model.named_parameters() if 'alpha' not in n and 'adaptive_span' not in n], lr=0.0001)
    attn_modules = [
        layer.attention.self
        for layer in model.TransfoXLModel.encoder.layer
        if isinstance(layer.attention.self, MixAdaptiveAttention)
    ]
    alpha_optimizers = [
        AdamW([attn.alpha], lr=0.1)
        for attn in attn_modules
    ]
    span_optimizers = [
        AdamW(attn.adaptive_span.parameters(), lr=0.1)
        for attn in attn_modules
    ]

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    alpha_schedulers = [ReduceLROnPlateau(alpha_optimizer, mode='min', factor=0.1, patience=2, verbose=True) for
                        alpha_optimizer in alpha_optimizers]
    span_schedulers = [ReduceLROnPlateau(span_optimizer, mode='min', factor=0.1, patience=2, verbose=True) for
                       span_optimizer in span_optimizers]

    num_epochs = 20

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

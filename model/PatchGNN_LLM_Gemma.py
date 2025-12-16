from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Optional, List
import os, glob
import gc

# ============== 强制依赖检查 ==============
try:
    # 使用 AutoModel 自动适配 Gemma
    from transformers import AutoModel, AutoConfig
except ImportError:
    raise ImportError("必须安装 transformers! 请运行: pip install transformers")

try:
    from peft import LoraConfig, get_peft_model, TaskType 
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# ============================================================

def softmax(src: torch.Tensor, index: torch.Tensor):
    N = maybe_num_nodes(index)
    global_out = src - src.max()
    global_out = global_out.exp()
    global_out_sum = scatter(global_out, index, dim=0, dim_size=N, reduce='sum')[index]
    return global_out / (global_out_sum + 1e-16)

def _randperm_like(x: torch.Tensor, num_nodes: int, g: torch.Generator):
    perm = torch.randperm(num_nodes, generator=g, device=x.device)
    return perm[x]

def perturb_edge_index(src, dst, num_nodes, mode="normal", keep_prob=1.0, seed=None):
    if mode == "normal" and keep_prob >= 0.999: return src, dst
    if seed is None: seed = 114514
    g = torch.Generator(device=src.device); g.manual_seed(seed)
    if mode == "shuffle":
        src = _randperm_like(src, num_nodes, g)
        dst = _randperm_like(dst, num_nodes, g)
    elif mode == "self":
        src = dst.clone()
    if keep_prob < 0.999:
        m = torch.rand(src.numel(), generator=g, device=src.device) < keep_prob
        if m.sum() == 0: m[0] = True
        src = src[m]; dst = dst[m]
    return src, dst

class Intra_Inter_Patch_Graph_Layer(MessagePassing):
    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_input // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha

        self.w_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_k)) for _ in range(self.n_heads)])
        self.bias_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_k)) for _ in range(self.n_heads)])
        for p in self.w_k_list: nn.init.xavier_uniform_(p)
        for p in self.bias_k_list: nn.init.uniform_(p)

        self.w_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_q)) for _ in range(self.n_heads)])
        self.bias_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_q)) for _ in range(self.n_heads)])
        for p in self.w_q_list: nn.init.xavier_uniform_(p)
        for p in self.bias_q_list: nn.init.uniform_(p)

        self.w_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_e)) for _ in range(self.n_heads)])
        self.bias_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_e)) for _ in range(self.n_heads)])
        for p in self.w_v_list: nn.init.xavier_uniform_(p)
        for p in self.bias_v_list: nn.init.xavier_uniform_(p)

        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, x, edge_index, edge_value, time_nodes,
                edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)
        return self.propagate(edge_index, x=x, edges_temporal=edge_value,
                              edge_same_time_diff_var=edge_same_time_diff_var,
                              edge_diff_time_same_var=edge_diff_time_same_var,
                              edge_diff_time_diff_var=edge_diff_time_diff_var,
                              n_layer=n_layer, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal,
                edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        msgs = []
        for i in range(self.n_heads):
            w_k, b_k = self.w_k_list[i][n_layer], self.bias_k_list[i][n_layer]
            w_q, b_q = self.w_q_list[i][n_layer], self.bias_q_list[i][n_layer]
            w_v, b_v = self.w_v_list[i][n_layer], self.bias_v_list[i][n_layer]

            att = self.each_head_attention(x_j, w_k, b_k, w_q, b_q, x_i,
                                           edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var)
            att = att / self.d_sqrt
            att = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * att
            att_norm = softmax(att, edge_index_i)

            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j, w_v[0]) + b_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j, w_v[1]) + b_v[1])
            sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j, w_v[2]) + b_v[2])
            sender = sender_stdv + sender_dtsv + sender_dtdv
            msgs.append(att_norm * sender)
        return torch.cat(msgs, 1)

    def each_head_attention(self, x_j, w_k, b_k, w_q, b_q, x_i,
                            e_stdv, e_dtsv, e_dtdv):
        x_i = e_stdv*(torch.matmul(x_i, w_q[0])+b_q[0]) + e_dtsv*(torch.matmul(x_i, w_q[1])+b_q[1]) + e_dtdv*(torch.matmul(x_i, w_q[2])+b_q[2])
        sdr = e_stdv*(torch.matmul(x_j, w_k[0])+b_k[0]) + e_dtsv*(torch.matmul(x_j, w_k[1])+b_k[1]) + e_dtdv*(torch.matmul(x_j, w_k[2])+b_k[2])
        return torch.squeeze(torch.bmm(sdr.unsqueeze(1), x_i.unsqueeze(2)), 1)

    def update(self, aggr_out, residual):
        return self.res * residual + F.gelu(aggr_out)

class TSTokenizer(nn.Module):
    def __init__(self, d_model, max_scales=8, prior_dim=0,
                 num_prompts: int = 32, num_buckets: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_scales = max_scales
        self.scale_pos = nn.Parameter(torch.randn(max_scales, d_model) * 0.02)
        self.var_proj = nn.Linear(d_model, d_model)
        self.scale_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model + d_model, d_model)
        self.prior_proj = nn.Linear(prior_dim, d_model) if prior_dim > 0 else None
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.prompts = nn.Parameter(torch.randn(num_prompts, d_model) * 0.02)
        self.num_buckets = num_buckets
        self.bucket_emb = nn.Embedding(num_buckets, d_model)
        self.val_proj   = nn.Linear(1, d_model)

    def _discretize(self, z: torch.Tensor):
        nb = self.num_buckets
        idx = torch.floor((z + 5.0) / 10.0 * (nb - 1)).long()
        return idx.clamp_(0, nb - 1)

    def build(self, scale_nodes, var_emb, query_times, time_embed_fn,
              prior_feats=None, hist_vals=None, hist_times=None, z_eps=1e-6):
        B, N, D = var_emb.shape
        device = var_emb.device
        S = min(len(scale_nodes), self.max_scales)

        prompt_tokens = self.prompts.unsqueeze(0).expand(B, -1, -1)

        if (hist_vals is not None) and (hist_times is not None):
            mu = hist_vals.mean(dim=2, keepdim=True)
            sigma = hist_vals.std(dim=2, keepdim=True)
            z = (hist_vals - mu) / (sigma + z_eps)
            z = z.clamp_(-5.0, 5.0)

            idx = self._discretize(z)
            emb_disc = self.bucket_emb(idx.squeeze(-1))
            emb_cont = self.val_proj(z)
            te_hist  = time_embed_fn(hist_times)
            history_tokens = emb_disc + emb_cont + te_hist
            B_, N_, K_, D_ = history_tokens.shape
            history_tokens = history_tokens.reshape(B_, N_*K_, D_)
        else:
            history_tokens = torch.zeros(B, 0, D, device=device)

        if S > 0:
            toks = []
            for s in range(S):
                h = scale_nodes[s]
                toks.append(self.scale_proj(h) + self.var_proj(var_emb) + self.scale_pos[s].view(1,1,-1))
            scale_tokens = torch.stack(toks, dim=2).reshape(B, N*S, D)
        else:
            scale_tokens = torch.zeros(B, 0, D, device=device)

        if self.prior_proj is not None and prior_feats is not None:
            prior_tok = self.prior_proj(prior_feats).unsqueeze(1)
        else:
            prior_tok = torch.zeros(B, 0, D, device=device)

        Lp = query_times.shape[2]
        te_q = time_embed_fn(query_times)
        var_rep = var_emb.unsqueeze(2).expand(B, N, Lp, D)
        query_tokens = self.query_proj(torch.cat([var_rep, te_q], dim=-1)).reshape(B, N*Lp, D)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, prompt_tokens, history_tokens, scale_tokens, prior_tok, query_tokens], dim=1)
        attn_mask = torch.ones(tokens.shape[:2], device=device, dtype=torch.long)

        c0, c1 = 0, 1
        p0, p1 = c1, c1 + prompt_tokens.shape[1]
        h0, h1 = p1, p1 + history_tokens.shape[1]
        s0, s1 = h1, h1 + scale_tokens.shape[1]
        r0, r1 = s1, s1 + prior_tok.shape[1]
        q0, q1 = r1, r1 + query_tokens.shape[1]
        idx = {'cls': (c0, c1), 'prompt': (p0, p1), 'history': (h0, h1),
               'scale': (s0, s1), 'prior': (r0, r1), 'query': (q0, q1)}
        return tokens, attn_mask, idx

class GemmaAdapter(nn.Module):
    """
    Gemma 适配器：
    1. 强制禁用 device_map='auto'，避免多卡切分导致的 RuntimeError
    2. 自动检测 hidden_size (Gemma-1B 约为 2048, 7B 约为 3072)
    3. 支持 LoRA
    """
    def __init__(self,
                 model_name_or_path: str,
                 ts_dim: int = 512,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 dtype: torch.dtype = torch.bfloat16,
                 local_files_only: bool = True,
                 enable_grad_ckpt: bool = True,
                 **kwargs  # 吸收所有多余参数
                 ):
        super().__init__()
        
        # === 强制配置 ===
        # 不要用 auto 切分，否则会和 model.to() 冲突
        device_map = None 
        
        print(f"-------- Configuring GemmaAdapter (Path: {model_name_or_path}) --------")
        
        # 1. 检查路径
        if not os.path.exists(model_name_or_path):
             raise FileNotFoundError(f"Model path not found: {model_name_or_path}")

        # 2. 加载配置并检测 hidden_size
        config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=local_files_only, trust_remote_code=True)
        self.llm_hidden_size = getattr(config, "hidden_size", 2048)
        print(f"   >>> Detected Gemma Hidden Size: {self.llm_hidden_size}")
        
        # 3. 初始化投影层
        self.input_proj = nn.Linear(ts_dim, self.llm_hidden_size).to(dtype=dtype)

        # 4. 加载基础模型
        try:
            self.llm = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                trust_remote_code=True,
                local_files_only=local_files_only,
                device_map=device_map, # 强制为 None
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            raise RuntimeError(f"[Fatal] Gemma 模型加载失败: {e}")

        # 5. 梯度检查点与冻结
        if enable_grad_ckpt and hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
        
        # 修改开始：微调策略 (Fine-tuning Strategy)
        # ============================================================
        
        # 1. 梯度检查点 (保持不变)
        if enable_grad_ckpt and hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
        
        # 2. 先默认冻结所有参数 (Baseline)
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # 3. 【新功能】解冻最后 N 层
        # 你可以在这里设置要微调的层数，比如 2 层
        finetune_last_n_layers = 2  
        
        if finetune_last_n_layers > 0:
            # 自动寻找存放 Layers 的属性名
            layers = None
            if hasattr(self.llm, "layers"): 
                layers = self.llm.layers  # Llama, Gemma, Qwen (AutoModel)
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
                layers = self.llm.model.layers # Llama, Qwen (Some versions)
            elif hasattr(self.llm, "h"): 
                layers = self.llm.h       # GPT-2
            elif hasattr(self.llm, "transformer") and hasattr(self.llm.transformer, "h"):
                layers = self.llm.transformer.h # GPT-2 (Some versions)

            if layers is not None:
                total_layers = len(layers)
                print(f"   >>> [Fine-tune] Unfreezing the last {finetune_last_n_layers} / {total_layers} layers...")
                
                # 遍历最后 N 层，将 requires_grad 设为 True
                for i in range(total_layers - finetune_last_n_layers, total_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True

            else:
                print("   [Warning] Could not find 'layers' list. Model remains fully frozen.")

        # 4. 开启输入梯度 (兼容 LoRA/Gradient Checkpointing)
        self.llm.enable_input_require_grads()
        
        # ============================================================
        # 修改结束
        # ============================================================

        # 6. 应用 LoRA
        print(f"   >>> Applying LoRA (r={lora_r})...")
        if _HAS_PEFT:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                # Gemma 的 Linear 层命名与 Llama/Qwen 类似
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
                lora_dropout=lora_dropout,
                bias="none"
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
        else:
            print("   [Warning] PEFT not installed, skipping LoRA (Model is frozen).")
            
        print("-------- GemmaAdapter Ready --------")

    def forward(self, tokens, attn_mask, idx_ranges):
        # 1. 投影
        proj_dtype = self.input_proj.weight.dtype
        tokens = tokens.to(dtype=proj_dtype)
        hidden_in = self.input_proj(tokens)

        # 2. 稳健获取目标设备
        try:
            # 尝试获取模型第一层的设备
            if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
                ref_param = self.llm.base_model.model.layers[0].input_layernorm.weight
            elif hasattr(self.llm, "layers"):
                ref_param = self.llm.layers[0].input_layernorm.weight
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
                ref_param = self.llm.model.layers[0].input_layernorm.weight
            else:
                ref_param = next(self.llm.parameters())
        except:
            # 兜底：使用 input_proj 的设备
            ref_param = self.input_proj.weight

        target_device = ref_param.device
        
        # 3. 将输入移到正确设备
        hidden_in = hidden_in.to(device=target_device)
        attn_mask = attn_mask.to(device=target_device, dtype=torch.long)

        # 4. 前向传播
        out = self.llm(
            inputs_embeds=hidden_in,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = out.hidden_states[-1]
        
        q0, q1 = idx_ranges['query']
        return last_hidden[:, q0:q1, :]

class RegressionHead(nn.Module):
    def __init__(self, d_in, d_hidden, out_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(inplace=True),
            nn.Linear(d_hidden, out_dim)
        )
    def forward(self, q_hidden):
        return self.mlp(q_hidden)

def sliced_wasserstein_distance(x, y, num_projections=64):
    device = x.device
    d = x.shape[-1]
    proj = torch.randn(num_projections, d, device=device)
    proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-9)
    x_proj = x @ proj.T; y_proj = y @ proj.T
    x_sort, _ = torch.sort(x_proj, dim=0); y_sort, _ = torch.sort(y_proj, dim=0)
    return torch.mean((x_sort - y_sort) ** 2)

def coral_loss(x, y, eps=1e-5):
    cx = x - x.mean(dim=0, keepdim=True); cy = y - y.mean(dim=0, keepdim=True)
    cov_x = (cx.T @ cx) / (x.shape[0]-1 + eps)
    cov_y = (cy.T @ cy) / (y.shape[0]-1 + eps)
    return torch.mean((cov_x - cov_y)**2)

def temporal_rank_consistency(h_seq: List[torch.Tensor]):
    loss = 0.0
    for i in range(1, len(h_seq)):
        loss = loss + torch.mean((h_seq[i] - h_seq[i-1])**2)
    return loss

class LLM4GNN(nn.Module):
    def __init__(self, args, supports=None):
        super().__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)
        self.nodevec = nn.Embedding(self.N, d_model)
        self.relu = nn.ReLU()
        
        self.graph_mode = "normal"      
        self.edge_keep_prob = 1.0      
        self.graph_seed = 2025         

        self.hist_len = getattr(args, "hist_len", 24)
        self.num_buckets = getattr(args, "num_buckets", 1024)
        self.num_prompts = getattr(args, "num_prompts", 32)

        self.use_llm = getattr(args, "use_llm", False)
        self.use_lora = getattr(args, "use_lora", False)
        
        # 修改：默认路径为 Gemma
        self.llm_name = getattr(args, "llm_name", "/home/dragonfly/LLMs/gemma-3-1b-it")
        self.lora_r = getattr(args, "lora_r", 8)
        self.patch_loss_weights = getattr(args, "patch_loss_weights", (0.1, 0.05, 0.02))

        for _ in range(self.n_layer):
            self.gcs.append(Intra_Inter_Patch_Graph_Layer(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        if args.task == 'forecasting':
            self.decoder = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, 1)
            )
        else:
            d_static = args.d_static
            if d_static != 0:
                self.emb = nn.Linear(d_static, args.ndim)
                self.classifier = nn.Sequential(
                    nn.Linear(args.ndim * 2, 200), nn.ReLU(), nn.Linear(200, args.n_class))
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(args.ndim, 200), nn.ReLU(), nn.Linear(200, args.n_class))

        if self.use_llm:
            self.ts_tokenizer = TSTokenizer(
                d_model=self.hid_dim, max_scales=args.patch_layer, prior_dim=0,
                num_prompts=self.num_prompts, num_buckets=self.num_buckets
            )
            # 修改：实例化 GemmaAdapter
            self.gemma_adapter = GemmaAdapter(
                model_name_or_path=self.llm_name,
                ts_dim=self.hid_dim,           
                lora_r=self.lora_r, 
                lora_alpha=16, lora_dropout=0.05,
                dtype=torch.bfloat16,
                local_files_only=True, 
                enable_grad_ckpt=True
            )
            self.reg_head = RegressionHead(
                d_in=self.gemma_adapter.llm_hidden_size, # 动态获取真实维度
                d_hidden=self.hid_dim,
                out_dim=1
            )

    def LearnableTE(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time):
        B, N, M, L, D = x.shape
        cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
        cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

        cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
        Bm, Mm, NL, _ = cur_mask.shape
        
        mask2 = cur_mask.reshape(Bm*Mm, NL, 1).contiguous()
        mask2_cpu = mask2.float().cpu()
        adj_cpu = (mask2_cpu @ mask2_cpu.transpose(1, 2)) > 0.5
        bm_idx, i_idx, j_idx = adj_cpu.nonzero(as_tuple=True)
        del mask2_cpu, adj_cpu, mask2; gc.collect()

        BM = B * M
        device = x.device
        bm_idx = bm_idx.to(device=device, dtype=torch.long)
        i_idx  = i_idx.to(device=device, dtype=torch.long)
        j_idx  = j_idx.to(device=device, dtype=torch.long)

        source_nodes = bm_idx * NL + i_idx
        target_nodes = bm_idx * NL + j_idx
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        
        total_nodes = cur_x.shape[0]
        s, t = perturb_edge_index(
            src=edge_index[0], dst=edge_index[1],
            num_nodes=total_nodes,
            mode=self.graph_mode,
            keep_prob=self.edge_keep_prob,
            seed=self.graph_seed,
        )
        edge_index = torch.stack([s, t], dim=0)
        
        cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c').contiguous()
        edge_time = (cur_x_time[source_nodes] - cur_x_time[target_nodes]).squeeze(-1)
        
        edge_stdv = (edge_time == 0).float().unsqueeze(-1)
        
        src_var_idx = (source_nodes // L) % N
        tgt_var_idx = (target_nodes // L) % N
        edge_dtsv = (src_var_idx == tgt_var_idx).float().unsqueeze(-1)
        
        edge_dtdv = ((edge_stdv + edge_dtsv) == 0).float()
        
        edge_self = torch.where((edge_stdv + edge_dtsv) == 2)
        edge_stdv[edge_self] = 0.0

        for gc_layer in self.gcs:
            cur_x = gc_layer(cur_x, edge_index, edge_time, cur_x_time, 
                             edge_stdv, edge_dtsv, edge_dtdv, 0)
        x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

        scale_nodes = []
        if M > 1 and M % 2 != 0:
            x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
            mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            M = M + 1

        obs_num_per_patch = torch.sum(mask_X, dim=3)
        x_time_per_patch = torch.sum(x_time, dim=3)
        avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype, device=x.device), obs_num_per_patch)
        avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)
        time_te = self.LearnableTE(x_time)
        Q = torch.matmul(avg_te, self.w_q)
        K = torch.matmul(time_te, self.w_k)
        V = torch.matmul(x, self.w_v)
        attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        attention = torch.div(attention, Q.shape[-1] ** 0.5)
        attention[torch.where(mask_X == 0)] = -1e-10
        scale_attention = torch.softmax(attention, dim=-2)
        mask_X = (obs_num_per_patch > 0).float()
        x = torch.sum((V * scale_attention), dim=-2)    
        x_time = avg_x_time
        scale_nodes.append(x.mean(dim=2))               

        for n_layer in range(1, self.patch_layer):
            B_, N_, T_, D_ = x.shape
            cur_x = x.reshape(-1, D_)
            cur_x_time = x_time.reshape(-1, 1)
            
            patch_indices = torch.arange(T_).float().to(x.device)
            cur_patch_indices = patch_indices.view(1, 1, T_).expand(B_, N_, T_).reshape(B_, -1)
            missing_indices = torch.where(mask_X.reshape(B_, -1) == 0)

            pim1 = cur_patch_indices.unsqueeze(1).expand(B_, N_*T_, N_*T_)
            pim2 = cur_patch_indices.unsqueeze(-1).expand(B_, N_*T_, N_*T_)
            patch_interval = pim1 - pim2
            if len(missing_indices[0]) > 0:
                patch_interval[missing_indices[0], missing_indices[1]] = 0
                patch_interval[missing_indices[0], :, missing_indices[1]] = 0

            edge_ind = torch.where(torch.abs(patch_interval) == 1)
            source_nodes = (N_ * T_ * edge_ind[0] + edge_ind[1])
            target_nodes = (N_ * T_ * edge_ind[0] + edge_ind[2])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])
            
            total_nodes_coarse = B_ * N_ * T_
            s2, t2 = perturb_edge_index(
                src=edge_index[0], dst=edge_index[1],
                num_nodes=total_nodes_coarse,
                mode=self.graph_mode,
                keep_prob=self.edge_keep_prob,
                seed=self.graph_seed,
            )
            edge_index = torch.stack([s2, t2], dim=0)

            if edge_index.shape[1] > 0:
                edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])
                
                edge_stdv = (edge_time == 0).float().unsqueeze(-1)
                
                src_var_n = (source_nodes // T_) % N_
                tgt_var_n = (target_nodes // T_) % N_
                edge_dtsv = (src_var_n == tgt_var_n).float().unsqueeze(-1)
                
                edge_dtdv = ((edge_stdv + edge_dtsv) == 0).float()
                edge_self = torch.where((edge_stdv + edge_dtsv) == 2)
                edge_stdv[edge_self] = 0.0

                for gc_layer in self.gcs:
                    cur_x = gc_layer(cur_x, edge_index, edge_time, cur_x_time, 
                                     edge_stdv, edge_dtsv, edge_dtdv, n_layer)
            
            x = rearrange(cur_x, '(b n t) c -> b n t c', b=B_, n=N_, t=T_, c=D_)
            if T_ > 1 and T_ % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(-2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B_, N_, 1, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                T_ = T_ + 1

            x = x.view(B_, N_, T_ // 2, 2, D_)
            x_time = x_time.view(B_, N_, T_ // 2, 2, 1)
            mask_X = mask_X.view(B_, N_, T_ // 2, 2, 1)

            obs_num_per_patch = torch.sum(mask_X, dim=3)
            x_time_per_patch = torch.sum(x_time, dim=3)
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype, device=x.device), obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)
            time_te = self.LearnableTE(x_time)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e-10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)    
            x_time = avg_x_time
            scale_nodes.append(x.mean(dim=2))

        x_final = x.squeeze() if x.dim() == 4 else x
        return x_final, scale_nodes

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None, return_aux_losses=False):
        B, M, L_in, N = X.shape
        self.batch_size = B

        K = min(self.hist_len, L_in)
        x_hist_last = X[:, -1, :, :]
        t_hist_last = truth_time_steps[:, -1, :, :]
        x_hist_last = x_hist_last[:, -K:, :]
        t_hist_last = t_hist_last[:, -K:, :]
        hist_vals = x_hist_last.permute(0, 2, 1).unsqueeze(-1).contiguous()
        hist_times = t_hist_last.permute(0, 2, 1).unsqueeze(-1).contiguous()

        X_enc = X.permute(0, 3, 1, 2).unsqueeze(-1)
        X_enc = self.obs_enc(X_enc)
        truth_time_steps_enc = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)
        mask_enc = mask.permute(0, 3, 1, 2).unsqueeze(-1)
        te_his = self.LearnableTE(truth_time_steps_enc)
        var_emb_full = self.nodevec.weight.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        X_enc = self.relu(X_enc + var_emb_full + te_his)

        h_final, scale_nodes = self.IMTS_Model(X_enc, mask_enc, truth_time_steps_enc)
        L_pred = time_steps_to_predict.shape[-1]

        if not self.use_llm:
            h = h_final.unsqueeze(-2).repeat(1, 1, L_pred, 1)
            tpred = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
            te_pred = self.LearnableTE(tpred)
            h = torch.cat([h, te_pred], dim=-1)
            outputs = self.decoder(h).squeeze(-1).permute(0, 2, 1).unsqueeze(0)
            return (outputs, {}) if return_aux_losses else outputs

        var_emb_llm = h_final
        t_pred = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
        tokens, attn_mask, idx_ranges = self.ts_tokenizer.build(
            scale_nodes=scale_nodes, var_emb=var_emb_llm,
            query_times=t_pred, time_embed_fn=self.LearnableTE, prior_feats=None,
            hist_vals=hist_vals, hist_times=hist_times
        )

        q_hidden = self.gemma_adapter(tokens, attn_mask, idx_ranges)
        
        reg_first_param = next(self.reg_head.parameters())
        q_hidden = q_hidden.to(device=reg_first_param.device, dtype=reg_first_param.dtype)
        pred = self.reg_head(q_hidden)
        pred = pred.view(B, N, L_pred, 1).permute(0, 2, 1, 3).squeeze(-1)
        outputs = pred.unsqueeze(0)

        if not return_aux_losses:
            return outputs

        q_hidden_agg = q_hidden.view(B, N, L_pred, -1).mean(dim=2)
        h_final_up = F.linear(h_final, self.gemma_adapter.input_proj.weight, self.gemma_adapter.input_proj.bias)
        bx = h_final_up.reshape(-1, h_final_up.shape[-1])
        by = q_hidden_agg.reshape(-1, q_hidden_agg.shape[-1])
        w1 = sliced_wasserstein_distance(bx, by)
        cr = coral_loss(bx, by)
        tmp = temporal_rank_consistency(scale_nodes) if len(scale_nodes) > 1 else torch.tensor(0., device=bx.device)
        aux_losses = {"w1": w1, "coral": cr, "temporal": tmp}
        return outputs, aux_losses

    def classification(self, X, truth_time_steps, mask=None, P_static=None):
        B, M, L_in, N = X.shape
        self.batch_size = B
        X_enc = X.permute(0, 3, 1, 2).unsqueeze(-1)
        X_enc = self.obs_enc(X_enc)
        truth_time_steps_enc = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)
        mask_enc = mask.permute(0, 3, 1, 2).unsqueeze(-1)
        te_his = self.LearnableTE(truth_time_steps_enc)
        var_emb = self.nodevec.weight.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        X_enc = self.relu(X_enc + var_emb + te_his)
        h_final, _ = self.IMTS_Model(X_enc, mask_enc, truth_time_steps_enc)
        if P_static is not None:
            static_emb = self.emb(P_static)
            return self.classifier(torch.cat([torch.sum(h_final, dim=-1), static_emb], dim=-1))
        return self.classifier(torch.sum(h_final, dim=-1))
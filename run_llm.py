import os
import sys
sys.path.append("..")
import time
import datetime
import argparse
import numpy as np
from random import SystemRandom

# ----------------- Args -----------------
parser = argparse.ArgumentParser('ITS Forecasting')
parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--epoch', type=int, default=1000, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('--pred_window', type=int, default=1, help="number of hours (months for ushcn) as pred window")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('--load', type=str, default=None, help="ID to load; if None, run new exp.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset: physionet, mimic, ushcn, activity")
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='Hi-Patch', help="Model name")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Hidden dim of node embeddings")
parser.add_argument('--alpha', type=float, default=1, help="Proportion of Time decay")
parser.add_argument('--res', type=float, default=1, help="Res")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

# ===== 新增：LLM/LoRA/对齐损失 开关与权重 =====
parser.add_argument('--use_llm', action='store_true', help='Enable LLM path (Graph->Token->LLM->Head)')
parser.add_argument('--llm_name', type=str, default='meta-llama/Llama-2-7b-hf', help='HF model path or local dir')
parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank for LLM adapter')
parser.add_argument('--w1', type=float, default=0.0, help='weight for sliced-Wasserstein loss')
parser.add_argument('--coral', type=float, default=0.0, help='weight for CORAL loss')
parser.add_argument('--temp', type=float, default=0.0, help='weight for temporal-consistency loss')
parser.add_argument('--fp16_llm', action='store_true', help='Run LLM adapter in fp16 (default in adapter); no cast here')

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
CUDA_LAUNCH_BLOCKING = 1
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import math
import torch
import torch.optim as optim
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

import warnings
warnings.filterwarnings("ignore")

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from lib.evaluation import *
# ✅ 确保这里导入的是“整合了 LLM 的 Hi_Patch”
# 例如：from model.hi_patch_llm import Hi_Patch
# from model.LLM_GNN3 import LLM4GNN  # 微调LLM
# from model.LLM_GNN2 import LLM4GNN  # 微调LLM_plts
from model.LLM_GNN_pro import LLM4GNN  # 微调LLM_plt

file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()

print("PID, device:", args.PID, args.device)

# ----------------- helpers -----------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    else:
        return layer_of_patches(n_patch + 1)

def _try_build_aux_losses(model, batch_dict, aux_weights):
    """
    尝试从 batch_dict 中抽取必要张量，调用 model.forecasting(..., return_aux_losses=True)
    返回 (aux_loss_tensor, details_dict)；如失败，返回 (None, {}).
    这个函数是“尽力而为”，保证即使键不匹配也不会打断训练。
    """
    lam_w1, lam_coral, lam_temp = aux_weights
    if not (hasattr(model, 'forecasting') and (lam_w1 > 0 or lam_coral > 0 or lam_temp > 0)):
        return None, {}

    try:
        # —— 下面的键名是常见命名，你的 parse_datasets 若不同，请按你项目实际改三个键 —— #
        # X: [B, M, L_in, N]
        X = batch_dict.get("observed_data")          # 常见：observed_data / X
        if X is None:
            X = batch_dict["X"]
        # truth_time_steps: [B, M, L_in, N]
        truth_ts = batch_dict.get("observed_tp")     # 常见：observed_tp / truth_time_steps
        if truth_ts is None:
            truth_ts = batch_dict["truth_time_steps"]
        # mask: [B, M, L_in, N]
        mask = batch_dict.get("observed_mask")       # 常见：observed_mask / mask
        if mask is None:
            mask = batch_dict["mask"]
        # time_steps_to_predict: [B, L_pred]
        t_pred = batch_dict.get("time_steps_to_predict")
        if t_pred is None:
            # 若数据管线没给，按 pred_window 伪造一个相对时间（不影响对齐损失的梯度传播形态）
            B, M, L_in, N = X.shape
            t0 = torch.zeros(B, 1, device=X.device, dtype=X.dtype)
            step = torch.ones(B, args.pred_window, device=X.device, dtype=X.dtype) * 1.0
            t_pred = t0 + torch.cumsum(step, dim=1)

        # 前向拿到 aux
        out, aux = model.forecasting(t_pred, X.to(args.device), truth_ts.to(args.device), mask.to(args.device), return_aux_losses=True)
        if not isinstance(aux, dict) or len(aux) == 0:
            return None, {}

        loss_aux = 0.0
        details = {}
        if "w1" in aux and lam_w1 > 0:
            loss_aux = loss_aux + lam_w1 * aux["w1"]
            details["w1"] = aux["w1"].item() if torch.is_tensor(aux["w1"]) else float(aux["w1"])
        if "coral" in aux and lam_coral > 0:
            loss_aux = loss_aux + lam_coral * aux["coral"]
            details["coral"] = aux["coral"].item() if torch.is_tensor(aux["coral"]) else float(aux["coral"])
        if "temporal" in aux and lam_temp > 0:
            loss_aux = loss_aux + lam_temp * aux["temporal"]
            details["temporal"] = aux["temporal"].item() if torch.is_tensor(aux["temporal"]) else float(aux["temporal"])

        if isinstance(loss_aux, float) and loss_aux == 0.0:
            return None, {}
        return loss_aux, details

    except Exception as e:
        # 打印一次，避免影响训练
        # print(f"[aux-loss skip] {e}")
        return None, {}

# ----------------- main -----------------
if __name__ == '__main__':
    utils.setup_seed(args.seed)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random()*100000)

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    # Parse dataset & model
    data_obj = parse_datasets(args, patch_ts=True)
    input_dim = data_obj["input_dim"]

    # === Model setting ===
    args.ndim = input_dim
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    args.patch_layer = layer_of_patches(args.npatch)
    args.scale_patch_size = args.patch_size / (args.history + args.pred_window)
    args.task = 'forecasting'

    # 把新增参数挂到 args，供模型读取
    setattr(args, "use_llm", bool(args.use_llm))
    setattr(args, "llm_name", args.llm_name)
    setattr(args, "lora_r", int(args.lora_r))
    setattr(args, "patch_loss_weights", (args.w1, args.coral, args.temp))

    # model = Hi_Patch(args).to(args.device)
    model = LLM4GNN(args).to(args.device)

    # 仅训练需要梯度的参数（LLM 主干被冻结时不会出现在 param list 中）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # 可选：若想给不同模块不同 lr，这里可以拆 param group。当前先用一个组保持简单。
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.w_decay)

    print('model', model)
    print('parameters (trainable):', count_parameters(model))

    # ------------- logging -------------
    if(args.n < 12000):
        args.state = "debug"
        log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
    else:
        log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr_{}seed.log". \
            format(args.dataset, args.model, args.state, args.patch_size, args.stride, args.nlayer, args.lr, args.seed)

    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    num_batches = data_obj["n_train_batches"]
    print("n_train_batches:", num_batches)

    best_val_mse = np.inf
    best_iter = -1
    test_res = None

    # ------------- Train loop -------------
    for itr in range(args.epoch):
        st = time.time()
        model.train()

        # —— Train one epoch —— #
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

            # 主损失（沿用你的原实现）
            train_res = compute_all_losses(model, batch_dict)  # 里头会 forward，并计算 criterion
            loss = train_res["loss"]

            # 可选：叠加对齐/一致性损失（仅当 use_llm 且权重>0）
            if args.use_llm and (args.w1 > 0 or args.coral > 0 or args.temp > 0):
                aux_loss, aux_info = _try_build_aux_losses(model, batch_dict, (args.w1, args.coral, args.temp))
                if aux_loss is not None:
                    loss = loss + aux_loss
                    # 记录一次 aux 数值（可选）
                    if len(aux_info) > 0:
                        logger.info(f"[aux] " + ", ".join([f"{k}:{v:.5f}" for k,v in aux_info.items()]))

            loss.backward()
            optimizer.step()

        # —— Validation —— #
        model.eval()
        with torch.no_grad():
            val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])

            # —— Test on best —— #
            if (val_res["mse"] < best_val_mse):
                best_val_mse = val_res["mse"]
                best_iter = itr
                test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])

            logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
            logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
            logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%"
                        .format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
            if(test_res is not None):
                logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%"
                            .format(best_iter, test_res["loss"], test_res["mse"], test_res["rmse"], test_res["mae"], test_res["mape"]*100))
            logger.info("Time spent: {:.2f}s".format(time.time()-st))

        if(itr - best_iter >= args.patience):
            print("Exp has been early stopped!")
            sys.exit(0)

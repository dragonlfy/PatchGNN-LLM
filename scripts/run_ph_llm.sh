patience=10
gpu=0
seeds="1 2 3 4 5"

# 打开 LLM
USE_LLM=1
LLM_NAME="/home/dragonfly/LLMs/Llama-3.2-1B"   # ← 本地目录
LORA_R=8

# 对齐损失（若不要就全设为 0）
W1=0.10
CORAL=0.05
TEMP=0.02

EXTRA_ARGS=""
if [ "$USE_LLM" -eq 1 ]; then
  EXTRA_ARGS+=" --use_llm --llm_name ${LLM_NAME} --lora_r ${LORA_R} --w1 ${W1} --coral ${CORAL} --temp ${TEMP}"
fi

for seed in {1..5}
do
    python run_llm.py \
    --dataset physionet --state 'def' --history 24 \
    --patience $patience --batch_size 2 --lr 1e-3 \
    --patch_size 750 --stride 750 --nhead 1 --nlayer 3 \
    --hid_dim 64 --alpha 1 \
    --seed ${seed} --gpu ${gpu} \
    ${EXTRA_ARGS}
done


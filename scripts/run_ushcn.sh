# ### tPatchGNN ###
# patience=10
# gpu=3

# for seed in {1..5}
# do
    
#     python run_models.py \
#     --dataset ushcn --state 'def' --history 24 \
#     --patience $patience --batch_size 192 --lr 1e-3 \
#     --patch_size 2 --stride 2 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 32 \
#     --outlayer Linear --seed $seed --gpu $gpu

# done

patience=10
gpu=3

for seed in {1..5}
do
    python run_hi-patch.py \
    --dataset ushcn --state def --history 24 \
    --patience $patience --batch_size 128 --lr 1e-3 \
    --patch_size 1.5 --stride 1.5 --nhead 4  --nlayer 2 \
    --hid_dim 64 \
    --seed $seed --gpu $gpu --alpha 1
done

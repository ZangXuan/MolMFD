data_path=./data/pretrain/ligands
save_dir=./checkpoints/pretrain
logfile=${save_dir}/train.log
n_gpu=2
MASTER_PORT=$RANDOM
lr=1e-4
wd=1e-4
batch_size=64
update_freq=1
max_hop=32
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
masked_shortest_loss=10
masked_degree_loss=-5
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
kl_loss=0.0005
orthogonal_loss=0.0005
contrastive_loss=0.005
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000

mkdir -p ${save_dir}
cp $0 ${save_dir}


export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path  --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task unimol --loss unimol_MAE --arch unimol_MAE_padding  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple --max-hop $max_hop \
       --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 100 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-degree-loss $masked_degree_loss \
       --masked-dist-loss $masked_dist_loss --masked-shortest-loss $masked_shortest_loss \
       --kl-loss $kl_loss --orthogonal-loss $orthogonal_loss --contrastive-loss $contrastive_loss \
       --decoder-x-norm-loss $x_norm_loss --decoder-delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --encoder-x-norm-loss $x_norm_loss --encoder-delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --encoder-unmasked-tokens-only \
       --decoder-layers 5 --decoder-ffn-embed-dim 2048 --decoder-attention-heads 64 \
       --save-dir $save_dir  --only-polar $only_polar > ${logfile} 2>&1
data_path="./data/protein_ligand_binding_pose_prediction"  # replace to your data path
save_dir="./save_pose"  # replace to your save path
n_gpu=2
MASTER_PORT=10086
finetune_mol_model="./checkpoints/pretrain/checkpoint_best.pt"
finetune_pocket_model="./weights/pocket_checkpoint.pt"
lr=3e-4
batch_size=8
epoch=50
dropout=0.2
warmup=0.06
update_freq=1
dist_threshold=8.0
recycling=3

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task docking_pose --loss docking_pose --arch docking_pose  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
       --mol-pooler-dropout $dropout --pocket-pooler-dropout $dropout \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $save_dir/tsb \
       --log-interval 100 --log-format simple \
       --validate-interval 1 --keep-last-epochs 10 \
       --best-checkpoint-metric valid_loss  --patience 2000 --all-gather-list-size 2048000 \
       --finetune-mol-model $finetune_mol_model \
       --finetune-pocket-model $finetune_pocket_model \
       --dist-threshold $dist_threshold --recycling $recycling \
       --save-dir $save_dir \
       --find-unused-parameters
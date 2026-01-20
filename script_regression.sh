data_path=/home/zx/unimol/molecular_property_prediction # replace to your data path
# data_path=/home/zx/unimol/my_data
save_dir="./save_finetune/esol"  # replace to your save path
n_gpu=1
MASTER_PORT=10084
dict_name="dict.txt"
# weight_path="./weights/checkpoint.pt"  # replace to your ckpt path
weight_path="./checkpoints/pretrain/checkpoint_6_790000.pt"
task_name="esol"  # molecular property prediction task name 
task_num=1
loss_func="finetune_mse"
lr=5e-4
batch_size=256
epoch=200
dropout=0.4
weight_decay=0
warmup=0.06
local_batch_size=32
only_polar=0
conf_size=11
seed=0


if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ] || [ "$task_name" == "qm9_3d" ]; then
	metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

rm -rf ${save_dir}
mkdir -p ${save_dir}
mkdir -p ${save_dir}/tmp

export CUDA_VISIBLE_DEVICES=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid,test \
       --conf-size $conf_size \
       --num-workers 8 --ddp-backend=c10d \
       --dict-name $dict_name \
       --task mol_finetune --loss $loss_func --arch unimol_base  \
       --classification-head-name $task_name --num-classes $task_num \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $weight_decay \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric $metric --patience 20 \
       --save-dir $save_dir --only-polar $only_polar \
       --finetune-from-model $weight_path \
       
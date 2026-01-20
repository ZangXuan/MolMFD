data_path="./data/protein_ligand_binding_pose_prediction"  # replace to your data path
results_path="./infer_pose"  # replace to your results path
weight_path="./save_pose/checkpoint_best.pt"
batch_size=8
dist_threshold=8.0
recycling=3

export CUDA_VISIBLE_DEVICES=1
python ./unimol/infer.py --user-dir ./unimol $data_path --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task docking_pose --loss docking_pose --arch docking_pose \
       --path $weight_path \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --dist-threshold $dist_threshold --recycling $recycling \
       --log-interval 50 --log-format simple
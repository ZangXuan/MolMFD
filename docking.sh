nthreads=20  # Num of threads
predict_file="./infer_pose/save_pose_test.out.pkl"  # Your inference file dir
reference_file="./data/protein_ligand_binding_pose_prediction/test.lmdb"  # Your reference file dir
output_path="./data/protein_ligand_binding_pose_prediction"  # Docking results path

python ./unimol/utils/docking.py --nthreads $nthreads --predict-file $predict_file --reference-file $reference_file --output-path $output_path
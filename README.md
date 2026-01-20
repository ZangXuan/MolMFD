# Molecular representation learning via multimodal fusion and decoupling

Official implementation of paper: Molecular representation learning via multimodal fusion and decoupling (Information Fusion 2025) [[MolMFD](https://doi.org/10.1016/j.inffus.2025.103493)]

## Environment Setup

The code of MolMFD is built upon [[UniMol](https://github.com/deepmodeling/Uni-Mol/tree/main/unimol)] documentation, please refer to its dependencies.

## Pre-train

Download the pre-training dataset from [[Pretrain_data](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/pretrain/ligands.tar.gz)] and place it in the ./data/pretrain directory.

You can pretrain the model by
```
bash script_pretrain.sh
```

## Molecular Property Prediction

Download the downstream dataset from [[MPP_data](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz)] and place it in the ./data directory.

You can finetune the model for classification tasks by
```
bash script_classification.sh
```

You can finetune the model for regression tasks by
```
bash script_regression.sh
```

## Protein-ligand docking pose prediction

Download the downstream dataset from [[PLDPP_data](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/protein_ligand_binding_pose_prediction.tar.gz)] and place it in the ./data directory.
Download the pretrained pocket weights from [[Pocket_weights](https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt)] and place it in the ./weights directory, naming it 'pocket_checkpoint.pt'.

You can finetune the model for protein-ligand docking pose prediction tasks by
```
bash script_docking.sh
bash infer_docking.sh
bash docking.sh
```


Citation
--------

Please kindly cite this paper as follows. Thank you.
```
@article{zang2025molecular,
  title={Molecular representation learning via multimodal fusion and decoupling},
  author={Zang, Xuan and Zhang, Junjie and Tang, Buzhou},
  journal={Information Fusion},
  pages={103493},
  year={2025},
  publisher={Elsevier}
}
```

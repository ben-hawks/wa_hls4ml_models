# wa-hls4ml Surrogate Models

This repository contains surrogate models and utilities for the wa-hls4ml dataset.

## **Repository Structure**

### **Directories**
- `transformer/`: Contains transformer based model and utilities.
  - `data.py`: Data preprocessing scripts.
  - `model.py`: Transformer model definitions.
  - `train.py`: Training scripts for the transformer model.
  - `requirements.txt`: Dependencies for transformer workflow.

- `GNN/`: Contains Graph Neural Network (GNN) model and utilities.
  - `DatasetMay29Complete.py`: Dataset preparation for GNN model.
  - `Models.py`: GNN model definitions.
  - `training_scripts/`: Scripts for training the GNN model.
  - `requirements.txt`: Dependencies for GNN workflows.

- `dataset/`: Contains dataset preparation scripts.
  - `Dataset_to_csvs6_with_ii.py`: Converts datasets to CSV format.

- `resource_report_results/`: Both models retrained on post-synthesis resource labels: weights, logs, plots and a comparison with the paper.

### **Files**
- `5_26_requirements.txt`: Dependencies for the project.
- `.gitignore`: Specifies files and directories to ignore.

## Training on the HuggingFace dataset

1. Download the [wa-hls4ml](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml) splits into `wa-hls4ml/`. Pass one `--include` pattern per call:
   ```bash
   for split in train val test; do
     hf download fastmachinelearning/wa-hls4ml --repo-type dataset --local-dir wa-hls4ml --include "$split/*"
   done
   ```
2. Convert them to NumPy arrays (about 40 minutes). `--resource-key` selects the labels: `resource_report` (post-logic-synthesis, the default) or `hls_resource_report` (HLS estimates, as in the paper). After an interruption, rerun with `--skip-existing`.
   ```bash
   cd dataset
   python Dataset_to_csvs6_with_ii.py --hf-root ../wa-hls4ml \
       --output-dir output/hf_split_resource_report --resource-key resource_report
   ```
3. Train the transformer. `--resume new_results_plots/<run>` continues an interrupted run.
   ```bash
   cd transformer
   python run.py --data-dir ../dataset/output/hf_split_resource_report
   ```
4. Train the GNN. Rerun with `--resume` to continue an interrupted run.
   ```bash
   cd GNN
   python training_scripts/y_03_GAT_vanilla_bigboi.py \
       --data-dir ../dataset/output/hf_split_resource_report \
       --output-dir results/gat_vanilla_resource_report
   ```

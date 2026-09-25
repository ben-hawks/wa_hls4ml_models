"""
Load a pretrained GNN checkpoint (e.g. the .pth file from the `gnn-weights-v1`
release) and optionally evaluate it on the test split.

Works with both checkpoint formats written by the training scripts:
  - best_checkpoint.pth      (model_config has 'use_enhanced')
  - final_gatv2_model.pth    (model_config has 'model_type')

Usage:
    python load_pretrained.py path/to/checkpoint.pth
    python load_pretrained.py path/to/checkpoint.pth \
        --data-dir dataset/Full_dataset_processed_split \
        --stats results/normalization_stats_01.npy
"""
import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Models import FPGA_GNN, FPGA_GNN_GATv2, FPGA_GNN_GATv2_Enhanced


def load_checkpoint(checkpoint_path, map_location='cpu'):
    """Load a checkpoint dict written by the training scripts."""
    return torch.load(checkpoint_path, map_location=map_location)


def build_model_from_config(model_config):
    """Instantiate the model described by a checkpoint's 'model_config'."""
    model_type = model_config.get('model_type')
    if model_type is None:
        model_type = 'FPGA_GNN_GATv2_Enhanced' if model_config.get('use_enhanced') else 'FPGA_GNN_GATv2'

    common = dict(
        node_feature_dim=model_config['node_feature_dim'],
        num_targets=model_config['num_targets'],
        hidden_dim=model_config['hidden_dim'],
        num_gnn_layers=model_config['num_gnn_layers'],
        mlp_hidden_dim=model_config['mlp_hidden_dim'],
        dropout_rate=model_config['dropout_rate'],
    )

    # Constructor arguments not stored in model_config match the values
    # used by the training scripts.
    if model_type == 'FPGA_GNN_GATv2':
        return FPGA_GNN_GATv2(
            num_attention_heads=model_config['num_attention_heads'],
            concat_heads=True,
            residual_connections=True,
            **common
        )
    if model_type == 'FPGA_GNN_GATv2_Enhanced':
        return FPGA_GNN_GATv2_Enhanced(
            num_attention_heads=model_config['num_attention_heads'],
            use_edge_features=True,
            **common
        )
    if model_type == 'FPGA_GNN':
        return FPGA_GNN(**common)
    raise ValueError(f"Unknown model_type: {model_type}")


def load_pretrained_model(checkpoint_path, device='cpu'):
    """Return (model, checkpoint) with weights loaded and the model in eval mode."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model_from_config(checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_on_test_set(model, data_dir, stats_path, device='cpu', batch_size=1024):
    """Evaluate on the test split, reporting denormalized metrics."""
    from torch_geometric.loader import DataLoader
    from Dataset3LogNorm import FPGAGraphDataset
    from utils.Utils import calculate_metrics

    train_features = os.path.join(data_dir, 'train_features.npy')
    train_labels = os.path.join(data_dir, 'train_labels.npy')
    test_features = os.path.join(data_dir, 'test_features.npy')
    test_labels = os.path.join(data_dir, 'test_labels.npy')

    if stats_path and os.path.exists(stats_path):
        feature_means, feature_stds, label_means, label_stds, use_log, log_eps, log_shift = \
            FPGAGraphDataset.load_normalization_stats(stats_path)
        stats = (feature_means, feature_stds, label_means, label_stds)
    else:
        # Stats are computed from the training split, exactly as during training
        print("No normalization stats file given; computing them from the training split...")
        print("WARNING: results are only meaningful with the normalization stats the checkpoint "
              "was trained with; recomputed stats may not match them.")
        train_dataset = FPGAGraphDataset(train_features, train_labels, stats=None)
        stats = (train_dataset.feature_means, train_dataset.feature_stds,
                 train_dataset.label_means, train_dataset.label_stds)
        use_log, log_eps, log_shift = False, 1e-6, None
        if stats_path:
            train_dataset.save_normalization_stats(stats_path)

    test_dataset = FPGAGraphDataset(
        test_features, test_labels, stats=stats,
        use_log_transform=use_log, log_epsilon=log_eps,
        log_shift=log_shift if use_log else None
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu())
            targets.append(batch.y.squeeze(1).cpu())
    predictions = test_dataset.denormalize_labels(torch.cat(predictions, dim=0))
    targets = test_dataset.denormalize_labels(torch.cat(targets, dim=0))

    feature_names = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']
    return calculate_metrics(predictions, targets, feature_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pretrained GNN checkpoint.")
    parser.add_argument('checkpoint', help="Path to the .pth checkpoint")
    parser.add_argument('--data-dir', default=None,
                        help="Directory with {train,test}_{features,labels}.npy; evaluates on the test split if given")
    parser.add_argument('--stats', default=None,
                        help="Normalization stats .npy used during training (e.g. results/normalization_stats_01.npy); "
                             "if missing, recomputed from the training split and saved here")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model, checkpoint = load_pretrained_model(args.checkpoint, device=args.device)
    print(f"Loaded {type(model).__name__} from {args.checkpoint}")
    print(f"Model config: {checkpoint['model_config']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if 'best_val_loss' in checkpoint:
        print(f"Best val loss: {checkpoint['best_val_loss']:.6e}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")

    if args.data_dir:
        metrics = evaluate_on_test_set(model, args.data_dir, args.stats, device=args.device)
        print("\nTest Set Metrics (denormalized):")
        for metric, value in metrics['overall'].items():
            print(f"  {metric}: {value:.4f}")
        for feature, feature_metrics in metrics['per_feature'].items():
            print(f"\n{feature}:")
            for metric, value in feature_metrics.items():
                print(f"  {metric}: {value:.4f}")

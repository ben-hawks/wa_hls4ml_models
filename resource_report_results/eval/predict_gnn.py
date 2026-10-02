"""Write test-set predictions of a trained GNN: python predict_gnn.py final_gatv2_model.pth out.npz"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, REPO, save_predictions

sys.path.insert(0, os.path.join(REPO, 'GNN'))
from Dataset3LogNorm import create_dataloaders_from_split_data
from load_pretrained import load_pretrained_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p = lambda s: os.path.join(DATA, s)
stats = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gnn', 'normalization_stats_log.npy')
_, _, test_loader, _, _ = create_dataloaders_from_split_data(
    train_features_path=p('train_features.npy'), train_labels_path=p('train_labels.npy'),
    val_features_path=p('val_features.npy'), val_labels_path=p('val_labels.npy'),
    test_features_path=p('test_features.npy'), test_labels_path=p('test_labels.npy'),
    stats_load_path=stats, batch_size=256, num_workers=1, pin_memory=device.type == 'cuda',
    use_log_transform=True, log_epsilon=1e-6)
model, _ = load_pretrained_model(sys.argv[1], device=device)
preds, targets = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu())
        targets.append(batch.y.squeeze(1).cpu())
denorm = test_loader.dataset.denormalize_labels
save_predictions(denorm(torch.cat(targets)).numpy(), denorm(torch.cat(preds)).numpy(), sys.argv[2])

"""Write test-set predictions of a trained transformer: python predict_transformer.py model.pt out.npz"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, REPO, save_predictions

sys.path.insert(0, os.path.join(REPO, 'transformer'))
from GNN.Dataset2 import create_dataloaders_from_split_data
from model import TransformerRegressor
from train import test_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p = lambda s: os.path.join(DATA, s)
_, _, test_loader, _, _ = create_dataloaders_from_split_data(
    train_features_path=p('train_features.npy'), train_labels_path=p('train_labels.npy'),
    val_features_path=p('val_features.npy'), val_labels_path=p('val_labels.npy'),
    test_features_path=p('test_features.npy'), test_labels_path=p('test_labels.npy'),
    stats_load_path=None, batch_size=512, num_workers=1, pin_memory=device.type == 'cuda',
    mode='transformer', use_log_transform=True, log_epsilon=1e-6)
test_loader.dataset.mode = 'transformer'
model = TransformerRegressor().to(device)
model.load_state_dict(torch.load(sys.argv[1], map_location=device))
y_true, y_pred = test_model(model, test_loader, device)
denorm = test_loader.dataset.denormalize_labels
save_predictions(denorm(torch.tensor(y_true)).numpy(), denorm(torch.tensor(y_pred)).numpy(), sys.argv[2])

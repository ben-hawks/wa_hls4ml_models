"""Compare saved test predictions against Table 4 of the wa-hls4ml paper (arXiv:2511.05615).

Usage: python compare_to_paper.py ../gnn/test_predictions.npz ../transformer/test_predictions.npz

Predictions are capped at the largest training label, as the training/evaluation code does.
SMAPE is reported both on raw predictions and with BRAM rounded to half blocks and DSP to whole
DSPs, since SMAPE counts any non-zero prediction for a true 0 as a 200% error.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, NAMES

PAPER_ORDER = ['BRAM', 'DSP', 'FF', 'LUT', 'CYCLES', 'II']
# Table 4, test set, in PAPER_ORDER: (R2, SMAPE %)
PAPER = {
    'GNN': {
        'All': ([0.51, 0.89, 0.74, 0.73, 0.89, 0.91], [19.5, 15.1, 11.6, 11.4, 15.7, 13.4]),
        'Dense': ([-0.51, -0.74, 0.73, 0.73, 0.82, 0.91], [24.6, 23.9, 11.8, 11.6, 15.8, 13.4]),
        'Conv1D': ([0.69, 0.02, 0.95, 0.96, 0.97, 0.97], [33.7, 36.7, 7.7, 6.2, 11.0, 11.1]),
        'Conv2D': ([0.44, 0.51, 0.92, 0.95, 0.84, 0.88], [31.2, 34.8, 8.8, 6.7, 19.5, 18.5]),
    },
    'Transformer': {
        'All': ([0.39, 0.29, 0.72, 0.67, 0.95, 0.95], [14.1, 10.8, 2.9, 2.9, 10.1, 14.1]),
        'Dense': ([0.39, 0.29, 0.71, 0.67, 0.95, 0.91], [13.6, 10.2, 2.7, 2.7, 9.9, 14.1]),
        'Conv1D': ([0.77, 0.41, 0.97, 0.96, 0.96, 0.96], [29.9, 22.7, 7.2, 5.8, 10.4, 11.2]),
        'Conv2D': ([0.79, 0.55, 0.93, 0.96, 0.93, 0.93], [18.8, 25.3, 8.0, 6.8, 16.3, 16.9]),
    },
}


def r2(t, p):
    return 1 - ((t - p) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)


def smape(t, p):
    return (np.abs(p - t) / ((np.abs(p) + np.abs(t)) / 2 + 1e-8) * 100).mean(0)


def subsets(parts):
    X = np.load(os.path.join(DATA, 'test_features.npy'), mmap_mode='r')
    layer_type = np.asarray(X[:, :, 9])  # padded layers are -1
    conv1d = (layer_type == 2).any(1)
    conv2d = (layer_type == 3).any(1) & ~conv1d
    return {
        'All': np.ones(len(parts), bool),
        'Dense': ~(conv1d | conv2d),
        'Conv1D': conv1d,
        'Conv2D': conv2d,
        'All, xcu250 only': parts == 'xcu250-figd2104-2L-e',
    }


def main(gnn_path, transformer_path):
    label_max = np.load(os.path.join(DATA, 'train_labels.npy')).max(0)
    order = [NAMES.index(n) for n in PAPER_ORDER]
    for model, path in [('GNN', gnn_path), ('Transformer', transformer_path)]:
        d = np.load(path)
        t, p = d['y_true'], np.minimum(d['y_pred'], label_max)
        rounded = p.copy()
        rounded[:, NAMES.index('BRAM')] = np.round(rounded[:, NAMES.index('BRAM')] * 2) / 2
        rounded[:, NAMES.index('DSP')] = np.round(rounded[:, NAMES.index('DSP')])
        print(f'\n### {model}\n')
        print('| Test subset | n | Metric | ' + ' | '.join(PAPER_ORDER) + ' |')
        print('|---|---|---|' + '---|' * len(PAPER_ORDER))
        for name, m in subsets(d['target_part']).items():
            ours_r2, ours_sm, ours_rsm = r2(t[m], p[m])[order], smape(t[m], p[m])[order], smape(t[m], rounded[m])[order]
            rows = []
            if name in PAPER[model]:
                pr2, psm = PAPER[model][name]
                rows += [('R² (paper)', [f'{v:.2f}' for v in pr2]), ('SMAPE % (paper)', [f'{v:.1f}' for v in psm])]
            rows += [('R²', [f'{v:.2f}' for v in ours_r2]), ('SMAPE %', [f'{v:.1f}' for v in ours_sm]),
                     ('SMAPE %, rounded', [f'{v:.1f}' for v in ours_rsm])]
            for i, (metric, vals) in enumerate(rows):
                label = f'{name} | {m.sum()}' if i == 0 else ' | '
                print(f'| {label} | {metric} | ' + ' | '.join(vals) + ' |')


if __name__ == '__main__':
    main(*sys.argv[1:3])

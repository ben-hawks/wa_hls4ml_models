"""Shared helpers: test-set metadata and saving predictions in a model-independent format."""
import glob
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(REPO, 'dataset', 'output', 'hf_split_resource_report')
HF_ROOT = os.path.join(REPO, 'wa-hls4ml')
NAMES = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']


def test_target_parts():
    """FPGA part of every test model, in the order of test_labels.npy (same filtering as the converter)."""
    parts = []
    for f in sorted(glob.glob(os.path.join(HF_ROOT, 'test', '*.json'))):
        for x in json.load(open(f)):
            if isinstance(x, dict) and any((x.get('resource_report') or {}).get(k) for k in ['ff', 'lut', 'bram', 'dsp']):
                parts.append(x.get('target_part'))
    return np.array(parts)


def save_predictions(y_true, y_pred, out):
    """Save denormalized test predictions after checking they line up with test_labels.npy."""
    raw = np.load(os.path.join(DATA, 'test_labels.npy'))
    assert np.allclose(y_true, raw, rtol=1e-3, atol=1e-2), 'test order/denormalization mismatch'
    parts = test_target_parts()
    assert len(parts) == len(raw), 'target_part metadata does not line up with test_labels.npy'
    np.savez_compressed(out, y_true=raw, y_pred=y_pred, target_part=parts)
    print('saved', out)

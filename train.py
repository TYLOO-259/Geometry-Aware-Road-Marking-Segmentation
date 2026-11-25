#!/usr/bin/env python3
"""Simplified public training example, adapted from the research scripts.

Not the original experimental pipeline. Optional experimental strategies and
feature normalization are not included. The private network and prepared
model-ready inputs are required; this file is not independently runnable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from config import Config


def load_meta(root, cfg):
    # Retain the processed-data metadata interface used by the source script.
    root = Path(root)
    meta = json.loads((root / 'normalization.json').read_text(encoding='utf-8'))
    if int(meta['num_classes']) != cfg.num_classes or len(meta['features']) != 7:
        raise ValueError('Unexpected class count or input dimension')
    labels = tuple(meta['labels'][str(i)] for i in range(cfg.num_classes))
    if labels != cfg.label_names:
        raise ValueError('Unexpected class order; use the labels in config.py')
    if meta['geometry']['layers'] != list(cfg.levels) or meta['geometry']['K'] != cfg.num_neighbors:
        raise ValueError('Unexpected point hierarchy or neighborhood size')
    splits = meta['splits']
    seen = set()
    for role in ('train', 'val', 'test'):
        scenes = splits[role]
        if not scenes or len(set(scenes)) != len(scenes) or seen.intersection(scenes):
            raise ValueError('Empty, repeated, or overlapping scene assignments')
        if any(not s or Path(s).name != s or '__' in s or any(ch in s for ch in '*?[]') for s in scenes):
            raise ValueError('Use simple scene identifiers without __ or glob characters')
        seen.update(scenes)
    return meta


def find_blocks(root, meta, role):
    out = []
    for scene in meta['splits'][role]:
        blocks = sorted((Path(root) / role / 'velodyne').glob(scene + '__*.npy'))
        if not blocks:
            raise FileNotFoundError(f'No blocks for {scene} in {role}')
        out.extend((role, p.stem) for p in blocks)
    return out


def label_mask(root, item, cfg):
    role, stem = item
    base = Path(root) / role
    raw = np.load(base / 'labels' / f'{stem}.npy').reshape(-1)
    mask = np.load(base / 'valid_mask' / f'{stem}.npy').reshape(-1)
    if raw.shape != (cfg.num_points,) or mask.shape != raw.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError(f'Invalid labels or mask: {stem}')
    valid = mask.astype(bool)
    if not valid.any() or not np.isin(raw[valid], np.arange(cfg.num_classes)).all():
        raise ValueError(f'Invalid valid-point labels: {stem}')
    y = np.full(raw.shape, cfg.ignore_label, dtype=np.int32)
    y[valid] = raw[valid].astype(np.int32)
    return y, valid


def class_counts(root, items, cfg):
    counts = np.zeros(cfg.num_classes, dtype=np.int64)
    for item in items:
        y, valid = label_mask(root, item, cfg)
        counts += np.bincount(y[valid], minlength=cfg.num_classes)
    return counts


def weights_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError('Invalid training class counts')
    frequencies = counts / counts.sum()
    # Small numerical safeguard for a class absent from the training set.
    weights = 1.0 / np.sqrt(np.maximum(frequencies, 1e-9))
    weights /= weights.mean()
    return np.clip(weights, *Config.weight_clip).astype(np.float32)


def learning_rate_for_epoch(epoch, epochs, base_lr, min_lr):
    if not 1 <= epoch <= epochs:
        raise ValueError('Epoch out of range')
    position = (epoch - 1) / max(epochs - 1, 1)
    return float(min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * position)))


def load_one(root, item, cfg):
    role, stem = item
    base = Path(root) / role
    # The example expects prepared, model-ready XYZRGBI, not raw LAS values.
    # It intentionally does not impose an additional normalization scheme.
    features = np.load(base / 'velodyne' / f'{stem}.npy').astype(np.float32)
    if features.shape != (cfg.num_points, cfg.in_features_dim) or not np.isfinite(features).all():
        raise ValueError(f'Invalid input block: {stem}')
    y, _ = label_mask(root, item, cfg)
    xyz = [features[:, :3]]
    neighbors, subs, interps = [], [], []
    for level, n in enumerate(cfg.levels):
        name = f'{stem}_layer{level}.npy'
        near = np.load(base / 'neigh_idx' / name).astype(np.int32)
        if near.shape != (n, cfg.num_neighbors) or near.min() < 0 or near.max() >= n:
            raise ValueError(f'Invalid KNN: {name}')
        neighbors.append(near)
        if level + 1 < cfg.num_layers:
            n_next = cfg.levels[level + 1]
            sub = np.load(base / 'sub_idx' / name).astype(np.int32).reshape(-1)
            interp = np.load(base / 'interp_idx' / name).astype(np.int32).reshape(-1, 1)
            if sub.shape != (n_next,) or sub.min() < 0 or sub.max() >= n:
                raise ValueError(f'Invalid sampling indices: {name}')
            if interp.shape != (n, 1) or interp.min() < 0 or interp.max() >= n_next:
                raise ValueError(f'Invalid interpolation indices: {name}')
            subs.append(sub)
            interps.append(interp)
            xyz.append(xyz[-1][sub])
    return dict(xyz=tuple(xyz), features=features, neigh_idx=tuple(neighbors),
                sub_idx=tuple(subs), interp_idx=tuple(interps)), y


def dataset(tf, root, items, cfg, shuffle):
    levels = cfg.levels
    signature = ({
        'xyz': tuple(tf.TensorSpec((n, 3), tf.float32) for n in levels),
        'features': tf.TensorSpec((cfg.num_points, cfg.in_features_dim), tf.float32),
        'neigh_idx': tuple(tf.TensorSpec((n, cfg.num_neighbors), tf.int32) for n in levels),
        'sub_idx': tuple(tf.TensorSpec((n,), tf.int32) for n in levels[1:]),
        'interp_idx': tuple(tf.TensorSpec((n, 1), tf.int32) for n in levels[:-1]),
    }, tf.TensorSpec((cfg.num_points,), tf.int32))
    def generate():
        order = np.random.permutation(len(items)) if shuffle else range(len(items))
        for i in order:
            yield load_one(root, items[int(i)], cfg)
    return tf.data.Dataset.from_generator(generate, output_signature=signature).batch(
        cfg.batch_size, drop_remainder=False).prefetch(2)


def make_model(cfg, method):
    try:
        from model_module import Config as NetworkConfig, RandLANetComplete
    except ImportError as exc:
        raise RuntimeError('The private network implementation (model_module.py) is not included.') from exc
    modes = {'baseline': ('off', False), 'gpe': ('locse', False),
             'gpe_gap': ('both', False), 'gpe_gap_samfp': ('both', True)}
    mode, use_samfp = modes[method]
    c = NetworkConfig()
    c.num_classes = cfg.num_classes
    c.in_features_dim = cfg.in_features_dim
    c.num_layers = cfg.num_layers
    c.d_out = list(cfg.d_out)
    c.conv_type = 'randla'
    c.use_scf = c.use_sdfm = c.use_dualstream = c.use_mfp_net = False
    c.use_gam = mode != 'off'
    c.gam_mode = mode
    c.use_samfp = use_samfp
    c.dropout_rate = cfg.dropout_rate
    return RandLANetComplete(c)


def metrics(conf):
    tp = np.diag(conf).astype(np.float64)
    union = conf.sum(1) + conf.sum(0) - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    return {'oa': float(tp.sum() / max(conf.sum(), 1)), 'miou': float(iou.mean()),
            'iou': iou.tolist(), 'valid_points': int(conf.sum())}


def update_conf(conf, y, logits):
    nc = conf.shape[0]
    y = np.asarray(y).reshape(-1)
    pred = np.asarray(logits).argmax(axis=-1).reshape(-1)
    valid = (y >= 0) & (y < nc)
    conf += np.bincount(nc * y[valid] + pred[valid], minlength=nc * nc).reshape(nc, nc)


def run(args):
    import tensorflow as tf
    cfg = Config(args.profile)
    meta = load_meta(args.data_root, cfg)
    train_items = find_blocks(args.data_root, meta, 'train')
    val_items = find_blocks(args.data_root, meta, 'val')
    model = make_model(cfg, args.method)
    weights = tf.constant(weights_from_counts(class_counts(args.data_root, train_items, cfg)))
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate,
                                         beta_1=cfg.beta_1, beta_2=cfg.beta_2)

    def focal_loss(y, logits):
        valid = (y >= 0) & (y < cfg.num_classes)
        safe = tf.where(valid, y, tf.zeros_like(y))
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=safe, logits=logits)
        term = ce * tf.pow(1.0 - tf.exp(-ce), cfg.focal_gamma) * tf.gather(weights, safe)
        return tf.reduce_sum(tf.where(valid, term, tf.zeros_like(term))) / tf.maximum(
            tf.reduce_sum(tf.cast(valid, tf.float32)), 1.0)

    @tf.function(reduce_retracing=True)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = tf.cast(model(x, training=True), tf.float32)
            loss = focal_loss(y, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients((g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None)
        return loss, logits

    @tf.function(reduce_retracing=True)
    def val_step(x, y):
        logits = tf.cast(model(x, training=False), tf.float32)
        return focal_loss(y, logits), logits

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    best = -1.0
    with (out / 'epoch_metrics.csv').open('w', newline='', encoding='utf-8') as log:
        writer = csv.writer(log)
        writer.writerow(['epoch', 'learning_rate', 'train_loss', 'train_miou', 'train_oa',
                         'val_loss', 'val_miou', 'val_oa'])
        for epoch in range(1, cfg.max_epoch + 1):
            lr = learning_rate_for_epoch(epoch, cfg.max_epoch, cfg.learning_rate, cfg.min_learning_rate)
            optimizer.learning_rate.assign(lr)
            results = []
            for items, step, shuffle in ((train_items, train_step, True), (val_items, val_step, False)):
                conf = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
                loss_total, n = 0.0, 0
                for x, y in dataset(tf, args.data_root, items, cfg, shuffle):
                    loss, logits = step(x, y)
                    loss_total += float(loss.numpy())
                    n += 1
                    update_conf(conf, y.numpy(), logits.numpy())
                m = metrics(conf)
                results.append((loss_total / max(n, 1), m))
            (tl, tr), (vl, va) = results
            writer.writerow([epoch, lr, tl, tr['miou'], tr['oa'], vl, va['miou'], va['oa']])
            log.flush()
            print(f'Epoch {epoch}/{cfg.max_epoch}: validation mIoU={va["miou"]:.4f}', flush=True)
            if va['miou'] > best:
                best = va['miou']
                model.save_weights(str(out / 'best.weights.h5'))
                (out / 'best_epoch.json').write_text(json.dumps({'epoch': epoch, 'validation': va}, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description='Simplified public training example')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--profile', choices=list(Config.profiles), default='road8')
    p.add_argument('--method', choices=['baseline', 'gpe', 'gpe_gap', 'gpe_gap_samfp'], default='gpe_gap_samfp')
    p.add_argument('--output-dir', type=Path, default=Path('runs/experiment'))
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Export raw XYZRGBI arrays (not normalized model inputs) to LAS for visualization."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def parse_scene_block(stem):
    match = re.fullmatch(r'(.+?)__(\d+)', stem)
    if match is None:
        raise ValueError(f'Expected scene_id__block_number.npy: {stem}')
    return match.group(1), int(match.group(2))


def load_block(path, label_path, mask_path):
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] != 7 or not np.isfinite(points).all():
        raise ValueError(f'Expected finite N x 7 XYZRGBI: {path.name}')
    if not np.isin(points[:, 3:6], np.arange(256)).all():
        raise ValueError('Expected raw integer RGB in 0..255')
    intensity = points[:, 6]
    if np.any((intensity < 0) | (intensity > 65535) | (intensity != np.rint(intensity))):
        raise ValueError('Expected raw integer intensity in 0..65535')
    labels = np.load(label_path).reshape(-1) if label_path.is_file() else None
    if labels is not None and len(labels) != len(points):
        raise ValueError(f'Label length mismatch: {path.name}')
    if mask_path.is_file():
        mask = np.load(mask_path).reshape(-1)
        if len(mask) != len(points) or not np.isin(mask, [0, 1]).all():
            raise ValueError(f'Invalid valid-point mask: {path.name}')
        keep = mask.astype(bool)
        points = points[keep]
        if labels is not None:
            labels = labels[keep]
    else:
        print(f'[WARNING] No valid-point mask for {path.name}; padding cannot be removed.')
    if len(points) == 0:
        raise ValueError(f'No valid points: {path.name}')
    if labels is not None and not np.isin(labels, np.arange(8)).all():
        raise ValueError('Expected compact semantic labels in 0..7 after masking')
    return points, labels


def write_las(path, points, labels=None, decimals=3):
    import laspy
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    scale = 10.0 ** (-decimals)
    offsets = xyz.min(axis=0)
    if np.any(np.rint((xyz - offsets) / scale) > np.iinfo(np.int32).max):
        raise ValueError('Coordinate extent exceeds LAS integer range; use fewer scale decimals')
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = np.full(3, scale)
    header.offsets = offsets
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz.T
    las.red, las.green, las.blue = (points[:, 3:6].astype(np.uint16) * 257).T
    las.intensity = points[:, 6].astype(np.uint16)
    if labels is not None:
        las.classification = labels.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    las.write(str(path))
    print(f'[OK] {path}: {len(points)} points')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True, help='Directory containing raw XYZRGBI NPY blocks')
    p.add_argument('--label-dir', type=Path, default=None)
    p.add_argument('--mask-dir', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=Path('output_las'))
    p.add_argument('--mode', choices=['block', 'scene'], default='block')
    p.add_argument('--scene', default=None, help='Optional generic scene identifier')
    p.add_argument('--scale-decimals', type=int, default=3)
    a = p.parse_args()
    if not 0 <= a.scale_decimals <= 9:
        p.error('scale-decimals must be between 0 and 9')
    label_dir = a.label_dir or a.input_dir.parent / 'labels'
    mask_dir = a.mask_dir or a.input_dir.parent / 'valid_mask'
    scenes = {}
    for path in sorted(a.input_dir.glob('*.npy')):
        scene, number = parse_scene_block(path.stem)
        if a.scene is None or scene == a.scene:
            scenes.setdefault(scene, []).append((number, path))
    if not scenes:
        raise FileNotFoundError('No matching point blocks')
    for scene, blocks in sorted(scenes.items()):
        if len({number for number, _ in blocks}) != len(blocks):
            raise ValueError(f'Duplicate block numbers: {scene}')
        parts, label_parts = [], []
        for _, path in sorted(blocks):
            points, labels = load_block(path, label_dir / path.name, mask_dir / path.name)
            if a.mode == 'block':
                write_las(a.output_dir / f'{path.stem}.las', points, labels, a.scale_decimals)
            else:
                parts.append(points)
                label_parts.append(labels)
        if a.mode == 'scene':
            if any(x is None for x in label_parts) and not all(x is None for x in label_parts):
                raise ValueError(f'Mixed labeled and unlabeled blocks: {scene}')
            labels = None if all(x is None for x in label_parts) else np.concatenate(label_parts)
            write_las(a.output_dir / f'{scene}.las', np.concatenate(parts), labels, a.scale_decimals)


if __name__ == '__main__':
    main()

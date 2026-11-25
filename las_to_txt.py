#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

LABEL_NAMES = {
    'road8': ['road', 'arrow', 'lane', 'dashed', 'zebra', 'text', 'diamond', 'other'],
    'toronto6': ['road', 'arrow', 'lane', 'dashed', 'zebra', 'other'],
}


def require_laspy():
    try:
        import laspy
    except ImportError as exc:
        raise SystemExit('laspy is required: python -m pip install laspy numpy') from exc
    return laspy


def fields(header):
    return list(header.point_format.dimension_names)


def choose_rgb_mode(src, chunk_size, preferred):
    if preferred != 'auto':
        return preferred
    laspy = require_laspy()
    maximum = 0
    multiples = True
    with laspy.open(str(src)) as reader:
        for pts in reader.chunk_iterator(chunk_size):
            for ch in ('red', 'green', 'blue'):
                a = np.asarray(getattr(pts, ch))
                if a.size:
                    maximum = max(maximum, int(a.max()))
                    multiples &= not np.any(a % 256 != 0)
    if maximum <= 255:
        return 'direct8'
    if multiples:
        return 'div256'
    raise ValueError(
        f'{src.name}: 16-bit RGB values are not all multiples of 256. '
        'Please verify encoding and specify --rgb-mode div257 or div256 explicitly; '
        'automatic guessing is refused.'
    )


def rgb_convert(pts, mode):
    raw = np.column_stack(
        [np.asarray(getattr(pts, ch), dtype=np.uint16) for ch in ('red', 'green', 'blue')]
    )
    if mode == 'direct8':
        if np.any(raw > 255):
            raise ValueError('RGB exceeds 255; direct8 is not allowed')
        return raw.astype(np.uint8)
    if mode == 'div256':
        if np.any(raw % 256):
            raise ValueError('RGB values are not multiples of 256; verify RGB encoding')
        return (raw // 256).astype(np.uint8)
    if mode == 'div257':
        return np.rint(raw.astype(np.float32) / 257).astype(np.uint8)
    raise ValueError(mode)


def extract_labels(pts, field, profile, label_base, invalid_policy, toronto_label_format="source"):
    available = fields(pts)
    if field not in available:
        raise ValueError(f'Missing label field {field!r}, available: {available}')
    raw = np.asarray(pts[field], dtype=np.float64)
    keep = np.isfinite(raw)
    invalid = int((~keep).sum())
    if invalid and invalid_policy == 'error':
        raise ValueError(
            f'{field} contains {invalid:,} NaN/Inf labels. '
            'After reviewing the audit report, you may explicitly use '
            '--invalid-label-policy drop'
        )
    finite = raw[keep]
    if np.any(finite != np.rint(finite)):
        raise ValueError(f'{field} contains finite non-integer labels; refusing to alter automatically')
    vals = finite.astype(np.int32)
    if profile == 'toronto6':
        allowed = [0, 1, 2, 3, 4, 7] if toronto_label_format == 'source' else list(range(6))
        if not np.isin(vals, allowed).all():
            raise ValueError(f'Toronto labels must be in {allowed}; got {np.unique(vals).tolist()}')
        mapped = np.where(vals == 7, 5, vals) if toronto_label_format == 'source' else vals
    elif label_base == 'zero':
        if np.any((vals < 0) | (vals > 7)):
            raise ValueError(f'Eight-class labels expected 0..7; got {np.unique(vals).tolist()}')
        mapped = vals
    else:
        if np.any((vals < 1) | (vals > 8)):
            raise ValueError(f'Explicit 1..8 mode; got {np.unique(vals).tolist()}')
        mapped = vals - 1
    return mapped.astype(np.uint8), keep


def export_one(src, out, args):
    laspy = require_laspy()
    dest = out / f'{src.stem}.txt'
    part = dest.with_suffix('.txt.part')
    excluded_path = out / f'{src.stem}__invalid_label_raw_indices.npy'
    if dest.exists() or part.exists() or excluded_path.exists():
        raise FileExistsError(
            f'Output or temporary file already exists: {dest} / {part} / {excluded_path} '
            '(will not overwrite)'
        )
    with laspy.open(str(src)) as reader:
        fs = fields(reader.header)
        if args.label_field not in fs:
            raise ValueError(f'{src}: missing label field {args.label_field!r}, available: {fs}')
        for key in ('red', 'green', 'blue', 'intensity'):
            if key not in fs:
                raise ValueError(f'{src}: missing {key}; XYZRGBI must be preserved')
        count = int(reader.header.point_count)
    mode = choose_rgb_mode(src, args.chunk_size, args.rgb_mode)
    print(f'[LAS] {src.name}: raw_points={count:,}, label={args.label_field}, RGB={mode}', flush=True)
    hist = np.zeros(len(LABEL_NAMES[args.profile]), np.int64)
    read_count = written = 0
    excluded = []
    try:
        with laspy.open(str(src)) as reader, part.open(
            'w', encoding='ascii', buffering=1 << 20, newline='\n'
        ) as f:
            for j, pts in enumerate(reader.chunk_iterator(args.chunk_size)):
                n = len(pts)
                xyz = np.column_stack(
                    [np.asarray(getattr(pts, k), dtype=np.float64) for k in ('x', 'y', 'z')]
                )
                if not np.isfinite(xyz).all():
                    raise ValueError(f'{src.name}: XYZ contains NaN/Inf')
                labels, keep = extract_labels(
                    pts, args.label_field, args.profile, args.label_base, args.invalid_label_policy,
                    args.toronto_label_format
                )
                if not keep.all():
                    excluded.append(read_count + np.flatnonzero(~keep).astype(np.int64))
                rgb = rgb_convert(pts, mode)
                intensity = np.asarray(pts.intensity, dtype=np.uint16)
                if len(labels):
                    table = np.column_stack((xyz[keep], rgb[keep], intensity[keep], labels))
                    np.savetxt(f, table, fmt=['%.6f'] * 3 + ['%d'] * 5, delimiter=' ')
                    written += len(labels)
                    hist += np.bincount(labels, minlength=len(hist))
                read_count += n
                if j % 10 == 0:
                    print(f'    read {read_count:,}/{count:,}, valid TXT rows {written:,}', flush=True)
        excluded_count = sum(len(a) for a in excluded)
        if read_count != count or written + excluded_count != count:
            raise ValueError(
                f'Point conservation error: read={read_count}, written={written}, '
                f'excluded={excluded_count}, raw={count}'
            )
        if excluded_count:
            np.save(excluded_path, np.concatenate(excluded))
        part.replace(dest)
    except BaseException:
        print(f'[INCOMPLETE] Conversion incomplete, temporary file kept for inspection: {part}', file=sys.stderr)
        raise
    print(f'[DONE] {src.stem}: TXT={written:,}, excluded nonfinite label={excluded_count:,}', flush=True)
    return {
        'scene': src.stem,
        'las': str(src),
        'txt': str(dest),
        'point_count': written,
        'las_point_count': count,
        'exported_valid_label_count': written,
        'excluded_nonfinite_label_count': excluded_count,
        'excluded_raw_indices_file': str(excluded_path) if excluded_count else None,
        'class_hist': hist.tolist(),
        'rgb_mode': mode,
        'label_field': args.label_field,
        'label_base': args.label_base,
        'toronto_label_format': args.toronto_label_format,
        'invalid_label_policy': args.invalid_label_policy,
        'xyz_precision': '%.6f',
        'point_order': 'retained original LAS order; omitted source rows recorded in sidecar',
    }


def inspect_one(src, args):
    laspy = require_laspy()
    with laspy.open(str(src)) as r:
        print(
            f'{src.name} | points={r.header.point_count:,} | '
            f'point_format={r.header.point_format.id} | fields={fields(r.header)}'
        )
        if args.label_field in fields(r.header):
            hist = {}
            for pts in r.chunk_iterator(args.chunk_size):
                a = np.asarray(pts[args.label_field], np.float64)
                u, c = np.unique(a, return_counts=True)
                for val, num in zip(u, c):
                    hist[str(val)] = hist.get(str(val), 0) + int(num)
            print('  LABEL HIST:', hist, flush=True)
        else:
            print(f'  [WARNING] {args.label_field!r} NOT FOUND', flush=True)


def main():
    p = argparse.ArgumentParser(description='Convert raw LAS to 8-column TXT, preserving original point order')
    p.add_argument('--input-dir', type=Path, default=Path('input_las'))
    p.add_argument('--out-dir', type=Path, default=Path('output_txt'))
    p.add_argument('--profile', choices=['road8', 'toronto6'], default='road8')
    p.add_argument('--label-field', default=None,
                   help='Default scalar field for eight classes, "class" for Toronto six classes')
    p.add_argument('--label-base', choices=['zero', 'one'], default='zero',
                   help='road8 default 0..7; specify "one" for explicit 1..8')
    p.add_argument('--toronto-label-format', choices=['source', 'compact'], default='source',
                   help='source: 0,1,2,3,4,7; compact: 0..5')
    p.add_argument('--rgb-mode', choices=['auto', 'direct8', 'div256', 'div257'], default='auto')
    p.add_argument('--chunk-size', type=int, default=200000)
    p.add_argument('--invalid-label-policy', choices=['error', 'drop'], default='error',
                   help='error: abort on NaN/Inf labels; drop: skip these points and save original LAS row indices')
    p.add_argument('--inspect', action='store_true', help='Only inspect fields and label distribution, no TXT written')
    p.add_argument('--names', nargs='*', default=None, help='Select file names (without .las), default all LAS')
    a = p.parse_args()
    if a.label_field is None:
        a.label_field = 'label' if a.profile == 'road8' else 'class'
    if a.chunk_size < 1:
        p.error('chunk-size must > 0')
    all_las = sorted(x for x in a.input_dir.glob('*.las') if x.is_file())
    if a.names:
        requested = set(a.names)
        srcs = [x for x in all_las if x.stem in requested]
        if {x.stem for x in srcs} != requested:
            p.error('Some LAS files specified by --names were not found')
    else:
        srcs = all_las
    if not srcs:
        raise FileNotFoundError(f'No LAS files found: {a.input_dir}')
    print(f'[{"INSPECT" if a.inspect else "EXPORT"}] files={len(srcs)}, profile={a.profile}, label_field={a.label_field}')
    if a.inspect:
        for src in srcs:
            inspect_one(src, a)
        return
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in srcs:
        rows.append(export_one(src, a.out_dir, a))
    report = {
        'version': '1.0',
        'generated_at': datetime.now().isoformat(),
        'profile': a.profile,
        'class_names': LABEL_NAMES[a.profile],
        'columns': ['X', 'Y', 'Z', 'R', 'G', 'B', 'Intensity', 'Label'],
        'features': 7,
        'classes': len(LABEL_NAMES[a.profile]),
        'files': rows,
        'total_las_points': sum(r['las_point_count'] for r in rows),
        'total_exported_points': sum(r['point_count'] for r in rows),
        'total_excluded_nonfinite_labels': sum(r['excluded_nonfinite_label_count'] for r in rows),
    }
    (a.out_dir / 'export_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    with (a.out_dir / 'export_counts.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(
            ['scene', 'valid_txt_rows', 'original_las_rows', 'excluded_nonfinite_labels']
            + [f'class{i}' for i in range(len(LABEL_NAMES[a.profile]))]
            + ['rgb_mode']
        )
        for d in rows:
            w.writerow(
                [
                    d['scene'],
                    d['point_count'],
                    d['las_point_count'],
                    d['excluded_nonfinite_label_count'],
                    *d['class_hist'],
                    d['rgb_mode'],
                ]
            )
    print('[COMPLETE]', a.out_dir, flush=True)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Verify the published source snapshot without weights, data or dependencies."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import os
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/publication/source_snapshot_manifest.json'
BANNED = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.onnx', '.joblib',
          '.pkl', '.pickle', '.npy', '.npz', '.ply', '.sens', '.pcd', '.so',
          '.a', '.o', '.pyc', '.jsonl', '.zip', '.tar', '.gz'}
CRITICAL = ['README.md', '新开对话阅读内容.md', 'CURRENT_EXPERIMENT_STATUS.md',
            '资料/当前基线修改方向.md', '资料/代码文件索引与当前主线地图.md',
            '资料/结果.md', 'full_bundle/README.md', 'sl_bundle/README.md',
            'research/geometry/README.md', 'docs/GITHUB_CURRENT_SOURCE.md']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    errors = []
    for relative, spec in manifest['files'].items():
        path = ROOT / relative
        if 'symlink' in spec:
            if not path.is_symlink() or os.readlink(str(path)) != spec['symlink']:
                errors.append('link changed: ' + relative)
            elif not path.exists() or not str(path.resolve()).startswith(str(ROOT) + '/'):
                errors.append('link escapes snapshot or is broken: ' + relative)
        elif not path.is_file() or path.is_symlink() or sha(path) != spec['sha256']:
            errors.append('missing or changed file: ' + relative)
        if path.suffix.lower() in BANNED:
            errors.append('excluded asset in manifest: ' + relative)
    index = ROOT / 'docs/cleanup/deep_active_cleanup_20261014/current_code_index.csv'
    with index.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = ROOT / row['path']
        if not path.is_file() or sha(path) != row['sha256']:
            errors.append('core code differs from local index: ' + row['path'])
    for folder in ['full_bundle/code', 'sl_bundle/code', 'research/geometry/code']:
        for path in (ROOT / folder).rglob('*.py'):
            try:
                ast.parse(path.read_text(encoding='utf-8'))
            except (SyntaxError, UnicodeError) as exc:
                errors.append('syntax: ' + str(path.relative_to(ROOT)) + ': ' + str(exc))
    for relative in CRITICAL:
        path = ROOT / relative
        for match in re.finditer(r'\[[^\]\n]*\]\(([^\)\n]+)\)', path.read_text(encoding='utf-8')):
            url = match.group(1).strip().strip('<>')
            if url.startswith(('https:', 'http:', '#', 'mailto:')):
                continue
            url = url.split('#', 1)[0]
            if url and not (path.parent / unquote(url)).exists():
                errors.append('broken reading link: ' + relative + ' -> ' + url)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps({'source_snapshot_verified': True,
                      'published_entries': len(manifest['files']),
                      'core_code_entries': len(rows),
                      'weights_and_data_required': False}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(verify())

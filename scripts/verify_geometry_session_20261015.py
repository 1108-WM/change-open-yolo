#!/usr/bin/env python3
"""Check published experiment records without loading GT or external assets."""
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'docs/publication/geometry_session_20261015'
TEXT_SUFFIXES = {'.json', '.csv', '.md', '.py', '.sh', '.log', '.txt'}


def verify():
    manifest = json.loads((META / 'local_source_manifest.json').read_text(encoding='utf-8'))
    errors = []
    total = 0
    for relative, spec in manifest['files'].items():
        path = ROOT / relative
        if (not path.is_file() or path.is_symlink()
                or ROOT not in path.resolve().parents):
            errors.append('missing or unsafe path: ' + relative)
            continue
        data = path.read_bytes()
        total += len(data)
        if len(data) != spec['bytes'] or hashlib.sha256(data).hexdigest() != spec['sha256']:
            errors.append('original record changed: ' + relative)
        if path.suffix not in TEXT_SUFFIXES or b'\x00' in data:
            errors.append('non-text asset: ' + relative)
    overview = ROOT / 'docs/SESSION_EXPERIMENTS_20261015.md'
    links = 0
    for match in re.finditer(r'\[[^\]\n]*\]\(([^\)\n]+)\)', overview.read_text(encoding='utf-8')):
        url = match.group(1).strip().strip('<>')
        if url.startswith(('https:', 'http:', '#', 'mailto:')):
            continue
        url = unquote(url.split('#', 1)[0])
        links += 1
        if url and not (overview.parent / url).is_file():
            errors.append('broken overview link: ' + url)
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(json.dumps({'geometry_session_verified': True,
                      'verbatim_files': len(manifest['files']),
                      'verbatim_bytes': total, 'overview_local_links': links,
                      'external_assets_required': False,
                      'experiments_rerun': False}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(verify())

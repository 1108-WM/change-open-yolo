"""Read-only lstat inventory; never follows symlinks or deletes project files."""
import os, json, time, heapq, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
started = time.time()
device = ROOT.stat().st_dev
seen = set()
dirs, links, errors, largest = {}, [], [], []
suffixes = collections.defaultdict(lambda: [0, 0, 0])
cache_dirs = []

def walk(path):
    rel = str(path.relative_to(ROOT))
    total = [0, 0, 0]  # allocated bytes, apparent bytes, regular file count
    try:
        st = path.lstat()
        total[0] = st.st_blocks * 512
        with os.scandir(path) as entries:
            for ent in entries:
                p = Path(ent.path)
                r = str(p.relative_to(ROOT))
                try:
                    s = ent.stat(follow_symlinks=False)
                    if ent.is_symlink():
                        target = os.readlink(p)
                        try:
                            ts = p.stat()
                            exists, target_dev = True, ts.st_dev
                        except OSError:
                            exists, target_dev = False, None
                        if r.count('/') < 3:
                            links.append(dict(path=r, target=target, exists=exists, target_device=target_dev))
                        total[0] += s.st_blocks * 512
                    elif ent.is_dir(follow_symlinks=False):
                        if s.st_dev != device:
                            errors.append(dict(path=r, reason='other filesystem skipped'))
                            continue
                        sub = walk(p)
                        total = [a+b for a,b in zip(total, sub)]
                    elif ent.is_file(follow_symlinks=False):
                        key = (s.st_dev, s.st_ino)
                        if s.st_nlink > 1 and key in seen:
                            continue
                        if s.st_nlink > 1:
                            seen.add(key)
                        alloc = s.st_blocks * 512
                        total[0] += alloc
                        total[1] += s.st_size
                        total[2] += 1
                        ext = ''.join(p.suffixes[-2:]) if p.suffix == '.gz' else p.suffix.lower()
                        v = suffixes[ext or '<none>']
                        v[0] += alloc; v[1] += s.st_size; v[2] += 1
                        item = (alloc, s.st_size, r, s.st_nlink, s.st_mtime)
                        if len(largest) < 600: heapq.heappush(largest, item)
                        elif item > largest[0]: heapq.heapreplace(largest, item)
                except OSError as exc:
                    errors.append(dict(path=r, reason=str(exc)))
    except OSError as exc:
        errors.append(dict(path=rel, reason=str(exc)))
    if rel.count('/') <= 2 or path.name in ('__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'):
        dirs[rel] = total
    if path.name in ('__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'):
        cache_dirs.append(dict(path=rel, allocated_bytes=total[0], files=total[2]))
    if path.parent == ROOT:
        print(f'{rel}: {total[0]/2**30:.3f} GiB ({total[2]} files)', flush=True)
    return total

total = walk(ROOT)
result = dict(root=str(ROOT), started=started, ended=time.time(), device=device,
              measurement='st_blocks*512; no symlink traversal; same device; hardlinks counted once',
              total=total, directories=dirs, links=links, errors=errors,
              suffixes=dict(suffixes), caches=cache_dirs,
              largest_files=[dict(allocated_bytes=a, apparent_bytes=b, path=p, nlink=n, mtime=m)
                             for a,b,p,n,m in sorted(largest, reverse=True)])
(OUT/'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(total=total, elapsed=time.time()-started, errors=len(errors)), ensure_ascii=False))

"""Measure gzip size and roundtrip hashes for historical relation tables; no archive/deletion."""
import csv,gzip,hashlib,json,zlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
rows=list(csv.DictReader(gzip.open(OUT/'files.csv.gz','rt')))
kinds={'stage_b_candidate_relations','frontend_hierarchy_relations','frontend_merge_relations'}
results=[]
for r in rows:
    if r['kind'] not in kinds:continue
    p=ROOT/r['path'];st=p.stat();assert st.st_size==int(r['bytes']) and st.st_mtime_ns==int(r['mtime_ns'])
    if r['kind']=='stage_b_candidate_relations':
        receipt=p.parent/'summary.json';expected=json.loads(receipt.read_text())['hashes']['relations']
    else:
        stage=p.parent.parent.name;scene=p.parent.name
        receipt=p.parents[2]/'stage_receipts'/(scene+'_'+stage+'.json')
        expected=json.loads(receipt.read_text())['files'][p.name]
    enc=zlib.compressobj(3,zlib.DEFLATED,31);dec=zlib.decompressobj(31)
    before=hashlib.sha256();after=hashlib.sha256();compressed=0;restored=0
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            before.update(block);chunk=enc.compress(block);compressed+=len(chunk)
            raw=dec.decompress(chunk);after.update(raw);restored+=len(raw)
    chunk=enc.flush();compressed+=len(chunk);raw=dec.decompress(chunk)+dec.flush();after.update(raw);restored+=len(raw)
    assert dec.eof and restored==st.st_size and before.hexdigest()==after.hexdigest()==expected,p
    assert p.stat().st_mtime_ns==st.st_mtime_ns
    results.append(dict(path=r['path'],kind=r['kind'],raw_bytes=st.st_size,allocated_bytes=st.st_blocks*512,
        gzip_level3_bytes=compressed,gzip_allocated_estimate=(compressed+4095)//4096*4096,
        original_sha256=expected,roundtrip_sha256=after.hexdigest(),
        scientific_receipt=str(receipt.relative_to(ROOT)),original_nlink=st.st_nlink,
        decision='候选：可恢复冷归档；本轮不压缩落盘、不删除原件；旧阶段入口需先恢复文件'))
    if len(results)%100==0:print('compression roundtrip verified',len(results),flush=True)
with (OUT/'archive_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
summary=dict(files=len(results),original_allocated_bytes=sum(r['allocated_bytes'] for r in results),
    gzip_bytes=sum(r['gzip_level3_bytes'] for r in results),
    gzip_allocated_estimate=sum(r['gzip_allocated_estimate'] for r in results),
    projected_reclaim_before_index_overhead=sum(r['allocated_bytes']-r['gzip_allocated_estimate'] for r in results if r['original_nlink']==1),
    all_roundtrip_and_scientific_receipt_hashes_passed=True,
    archives_created=0,source_files_deleted=0,compression_level=3,
    limitation='Compressed size measured in memory, not an archive installed or an already-safe deletion. Preserve both receipt paths and a restore-before-replay workflow.')
(OUT/'archive_measurement.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

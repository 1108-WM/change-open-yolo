from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
a=json.loads((OUT/'assets_verified.json').read_text())
p=ROOT/'新开对话阅读内容.md';t=p.read_text();start=t.index('**2026-10-14最新追加');end=t.index('\n> **语义',start)
latest='''**2026-10-14最新接续：Full/SL/几何集中整理和三轮清理已完成。** 先读[最新精简回执](docs/cleanup/deep_active_cleanup_20261014/README.md)、[当前代码地图](资料/代码文件索引与当前主线地图.md)，再按[Full](full_bundle/README.md)、[SL](sl_bundle/README.md)、[几何](research/geometry/README.md)入口阅读。用户已授权抽取公式后删除旧文件：本轮38个退役完整入口拆为29个精简公共模块，21,203行→4,778行，133个函数/类定义一致；26个旧实验测试删除，41个有效Full/SL测试归回各目录。42个旧.local根（GD15/GD20、Q29、smoke、早期覆盖诊断）归档结论与源码后删除，分配块9.248505GiB；本轮空闲净增约8.77GiB。当前磁盘空闲108.13GiB，保留50GiB后可开发约58.13GiB；三轮净增约56.55GiB。480测试及4子测试、281回放、9后端/7权重/champion、925发布hash、964路由/972帧/42删除项共18,369资产核对通过。未重训、未全量GPU重算、未启动研究。当前源码hash查最新索引，旧合同/seal保持历史身份，不恢复退役入口。前两轮[集中迁移](docs/cleanup/full_consolidation_execution_20261014/README.md)与[范围复查](docs/cleanup/full_scope_recheck_20261014/README.md)回执保留为各自执行时点。'''
t=t[:start]+latest+'\n'+t[end:];p.write_text(t)
p=ROOT/'CURRENT_EXPERIMENT_STATUS.md';t=p.read_text();mark='\n> **语义与选择';i=t.index(mark)
paragraph='''
**最新深度精简已完成。** 38个旧策略完整入口归档删除，现用定义集中为29个公共模块，21,203行减至4,778行；133个函数/类仅调整导入路径、定义一致。26个对应旧实验测试删除，41个有效Full/SL测试归回对应目录。42个GD15/GD20、Q29、单场景smoke、早期类别覆盖.local根归档后删除，分配块9.248505GiB，压缩历史约505MiB；本轮空闲净增约8.77GiB。当前空闲108.13GiB，保留50GiB后可开发约58.13GiB；三轮净增约56.55GiB，不与目录迁移体积重复相加。480测试/4子测试、281回放、9后端/7权重/champion、925发布hash、18,369资产核对通过，Full/SL输出一致；未重训或启动研究。Full tools当前92个顶层实现+29公共模块+包初始化，另有4个utils；Full/SL/几何测试分别35/17/28。最新[执行回执](docs/cleanup/deep_active_cleanup_20261014/README.md)和[文件/hash索引](docs/cleanup/deep_active_cleanup_20261014/current_code_index.csv)替代上文前两轮索引的当前身份；其余未确定候选仍保留，未宣称完全最小包。
'''
t=t[:i]+paragraph+t[i:];p.write_text(t)
# Current handoff hashes are separate from immutable historical seals.
paths=['新开对话阅读内容.md','CURRENT_EXPERIMENT_STATUS.md','资料/代码文件索引与当前主线地图.md','full_bundle/README.md','sl_bundle/README.md','research/geometry/README.md']
(OUT/'handoff_current_hashes.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},ensure_ascii=False,indent=2)+'\n')
(OUT/'delivery_verified.json').write_text(json.dumps(dict(complete=True,asset_checks=a['checks'],tests=480,subtests=4,replay_checks=281,formula_classes_functions=133,retired_code_entries=38,retired_tests=26,retired_local_roots=42,current_code_index='current_code_index.csv',independent_runtime_files=925,dv3_models=7,champion_loaded=True,scope='authorized code refactor and reviewed project-local retirement; external input assets retained'),ensure_ascii=False,indent=2)+'\n')
print('Current handoff and receipt updated')

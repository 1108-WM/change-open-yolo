#!/usr/bin/env python3
"""Scoped report/closure only, never writes historical handoffs or production."""
from pathlib import Path
from collections import Counter
import sys,time,json,shutil
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_query_relation_diagnostic as run
OUT=run.OUT;read=run.read;seal=run.seal;sha=run.sha

def main():
    c=read(OUT/'contract.json');assert read(OUT/'verification.json')['complete'] and read(OUT/'supply_verification.json')['complete'];run.diag.verify(read(OUT/'analysis_frozen.json')['outputs']);run.diag.verify(read(OUT/'supply_verification.json')['evidence_outputs'])
    summary=read(OUT/'analysis/summary.json');comparisons=read(OUT/'analysis/pair_comparisons.json');rc=read(OUT/'analysis/region_comparisons.json');bins=read(OUT/'analysis/category_bins.json');refs=[];parents=[]
    for s in c['scenes']:refs+=read(OUT/'evaluation'/s/'posthoc_supply.json');parents+=read(OUT/'relations'/s/'parents.json')
    graph=read(OUT/'relations_frozen.json')['counts'];rows=['# Clutt3R-Seg启发：query关系与补点供给诊断','', '按顺序在冻结下游AP报告完成后执行。仅借鉴论文III-B“先建立同实例关系，再决定组合”的思想；未复制包含树、Duoduo语义网络、supervoxel或父替换。原T父、halo、区域、冻结P1和最多六帧保持，未训练、未增加观测、未生成生产合并候选。','',f"全量旧fit12/exposed16与internal24共52物理场景，207个原采样T父；其中{summary['stable_parents']}稳定、{207-summary['stable_parents']}不稳定。每父固定原top1锚，query集合=父top2+每个父外整块均值top2，ID并列排序后去重。无GT图先封存：{graph['candidate_queries']}个候选query、{graph['regions']}块、{graph['query_region_edges']}条二部边、{graph['query_pairs']}条query对和{graph['frame_records']}条帧证据。未只选已知成功父或漏补块。",'', '52场景全部属于已暴露开发诊断，不能称盲测、独立验收或泛化结果。原holdout、历史结果及最小阅读入口未修改。','', '## 关系类别与必要负例','', '|后置局部关系|query对|物理场景|零重叠|无可比帧|有共同帧|有冲突帧|','|---|---:|---:|---:|---:|---:|---:|']
    for b in bins:rows.append(f"|{b['relation']}|{b['pairs']}|{b['scenes']}|{b['no_overlap']}|{b['no_valid_frames']}|{b['any_common']}|{b['any_conflict']}|")
    rows+=['','关系仅在父halo内、稳定父目标下标注。双方各有>=20支持点、已标注比例>=.9且目标已知纯度>=.9才标局部同目标；高重叠（描述性IoU>=.5）重复与低重叠互补分开。小支持、身份不稳和混杂分别保留，不能把unknown强行当邻物或安全。互补指双方独占部分均有目标点，不等于两者是恢复父目标的必要最小集合。','', '完整场景P1软/硬体积、halo与存储ROI内目标/其他/未知及ROI外未观测硬点分别记录。28个局部互补query对在存储ROI内仍包含8,065个其他已知点、ROI外共有154,313个未检查硬点（关系重复计数）；因此“局部同目标”不能证明全query取并集安全。没有完整场景逐query点集，不能作全局身份证明。','', '## 无GT字段是否有区分','', '以下都是开发诊断，不拟合模型、不选择新阈值。AUC方向保留原字段；小于.5表示该字段反向，而不是事后换符号后宣称成功。macro按同时有正负例的物理场景算，bootstrap以场景为单位200次。','', '|范围/字段|互补/邻物条目|整体AUC|场景macro AUC|95%场景bootstrap|同时有两类的场景|','|---|---:|---:|---:|---|---:|']
    wanted=[('anchor_pairs','hard_iou'),('anchor_pairs','minimum_distance'),('anchor_pairs','region_contact_edges'),('anchor_pairs','min_parent_coverage'),('anchor_pairs','view_direction'),('no_overlap','hard_iou'),('no_overlap','minimum_distance'),('no_overlap','view_direction')]
    for name,feature in wanted:
        r=next(r for r in comparisons if r['stratum']==name and r['feature']==feature);fmt=lambda v:'—' if v is None else f'{v:.4f}';rows.append(f"|{name}/{feature}|{r['positive']}/{r['negative']}|{fmt(r['auc'])}|{fmt(r['macro_scene_auc'])}|{fmt(r['bootstrap_p025'])}–{fmt(r['bootstrap_p975'])}|{r['scene_count_both_labels']}|")
    rows+=['','含锚的16条局部互补与159条邻物关系中，重叠、部分父支持和接触有区分信号。但全量28条互补中5条完全无重叠；在无重叠子集IoU必然AUC=.5，距离方向甚至与含锚总体相反。不能把高重叠过滤直接作为互补分组方案。重复query的独立比较另存，未让大量重复正例掩盖低重叠碎片。','', '在query块均值>=.5的二部边中，纯目标块968条、纯邻物块603条；`query_parent_mean`整体/场景macro AUC约.9388/.9327，`hard_parent_fraction`约.9389/.9311。局部共同帧约.5528/.5871，整父共同帧约.5671/.6063，未显示统一多帧区分。各条目高度相关，样本量应看场景，非11004条独立实验。全部混合、未标注块未用于纯目标/纯邻物二元AUC，仍在数据和类别统计保留。','', '局部帧参考固定为补点块最近原父成员区域中的锚query>=.5点，空参考不回退；这只是可复现的局部支持描述，未证明最优关联。每帧分别记录缺失/不可比、可比但无共同、明确冲突；query对使用独占支持比较，重复query没有独占支持时多为缺失，不能把缺失当反例。','', '## 信息供给与完整NT机会','', '下列**全部为GT后置参照**，用于解释信息供给，没有进入冻结推理或生产候选。原父保留，读出仅允许父外可编辑域补点；“单query”不要求该query单独重建整个父。纯整块union只是一个可行构造，非任意整块优化的理论上界。','', '|后置参照|完整NT增量>=.02目标|场景|连续IoU增量和|','|---|---:|---:|---:|']
    labels={'point_optimum':'同域目标点最优','pure_regions':'全部纯目标整块union','candidate_safe_regions':'候选query支持的纯目标块union','all150_safe_regions':'所有150 query支持的纯目标块union','best_candidate_single':'候选中最佳单query局部补点','best_all150_single':'全部150中最佳单query局部补点','best_candidate_anchor_pair':'候选最佳锚+query局部补点','all_candidate_union':'全部候选hard支持局部union（解释性）'}
    for k,label in labels.items():r=summary['posthoc_pool'][k];rows.append(f"|{label}|{r['pool_gain02']}|{len({x['scene'] for x in refs if x.get(k+'_gain02')}) if k not in ['all_candidate_union'] else sum(any(x.get(k+'_gain02') for x in refs if x['scene']==s) for s in c['scenes'])}|{r['pool_delta_sum']:.6f}|")
    # pool aggregation deduplicates targets. Use actual gain02 scenes from sealed target records.
    for k,label in labels.items():
        gain_scenes=set()
        for s in c['scenes']:
            if any(r['mode']==k and r['gain02'] for r in read(OUT/'evaluation'/s/'pool.json')):gain_scenes.add(s)
        for i,line in enumerate(rows):
            if line.startswith('|'+label+'|'):
                r=summary['posthoc_pool'][k];rows[i]=f"|{label}|{r['pool_gain02']}|{len(gain_scenes)}|{r['pool_delta_sum']:.6f}|"
    better=sum(r['best_candidate_anchor_pair_iou']>r['best_candidate_single_iou']+1e-12 for r in refs);unique=sum(r['best_candidate_anchor_pair_gain02'] and not r['best_candidate_single_gain02'] for r in refs)
    rows+=['',f'候选单query与全部150最佳单query均保留6场景/6目标>=.02机会；候选锚+query也是6场景/6目标。虽然{better}个稳定父的锚+query比最佳单query局部IoU更高，却有{unique}个“只有锚+query达到池增量>=.02”的父。当前不能以供给参照宣称多query分组比关联选择提供更多池机会。候选集合支持全部5个纯整块GT构造机会；只说明支持信息在集合内，不能说明无GT能挑出这些块。全部query取并集仅2个>=.02目标，不能拿它当分组机制。','', '|场景/父|原锚|最佳候选单query|较NT增量|说明|','|---|---:|---:|---:|---|']
    opportunities=read(OUT/'supplement/opportunity_evidence.json')
    for r in opportunities:
        if r['single_gain_over_NT']>=.02 or r['pure_gain_over_NT']>=.02:
            rows.append(f"|{r['scene']}/{r['candidate_id']}|{r['anchor']}|{r['best_candidate_query']}|{r['single_gain_over_NT']:+.6f}|{r['interpretation']}；纯整块参照{r['pure_gain_over_NT']:+.6f}|")
    rows+=['','关键失败例：','', '- scene0002/c006的25点块：q7支持缺块，但对父主体软均值近0、硬覆盖0；q7与锚q25无重叠，距离约.1166m；2帧可比均冲突。局部是同目标互补，但“高父支持/重叠/共同帧”限制会丢掉此真实机会。原父保留后单q7补点GT参照可获+.04074，不能据此声称q7独立解释完整父。','- scene0085/c007：q34/122/137支持多个漏块，父硬覆盖仅约.37/.37/.36，和锚q53重叠很低；query独占支持有1共同帧、5冲突帧，而最近局部父支持参考对这些漏块为空。存在+.03112单query参照，但统一多帧AND不能安全验证。','- scene0078/c004：已有多个支持父和漏块的重复query，却最佳单query/锚对补点仍分别比NT差约.01583/.01623；纯整块参照可+.0294。query硬读出引入其他区域，问题不是候选query集合漏供给，而是区域选择/混杂。','- scene0688/c006：单q55参照可+.04251，纯整块构造无机会，存储ROI硬mask含426未标注点；属于边界/未知约束，不能仅凭局部主身份宣称安全。','', '逐块“单query同时支持父与块”的半支持描述和“候选联合需要多query”的代理单独保存。其阈值仅用于诊断，不是证明必须多实例分组；例如父保留时，一个只支持缺块的query也可以形成单query补点机会。','', '## 本阶段判断','', '**有局部可分性及现有query供给，但尚无“可安全保留多个场景新增池价值”的无GT分组证据。** 不能判当前信息完全无身份能力，也不能宣布互补分组成立。高父支持、高重叠等强平均信号会丢掉关键低重叠机会，多帧冲突又同时出现在正例；局部query身份与全场景安全存在证据缺口。','', '停止在当前单父范围直接叠加实例树、网络、loss或全部query合并。若继续设计，优先单独研究query—区域关联：区分“可单query关联”“低重叠互补”“块内混杂/未知”和“halo外供给”；需在另一个未参与规则拟定的隔离场景集冻结验证。当前52场景不能作为新机制独立验收，本轮没有设计或运行新分组规则。','', '## 核验与产物','', '独立复算全部207候选集合、query—块支持、5707对局部/ROI GT点数、41450帧局部/整父/独占支持证据；另用暴力点距核查无重叠query距离。供给核验独立逐点计数重算1144父模式IoU和9256完整NT池目标模式。未重跑冻结Mask3D、原深度投影或构造全场景query点集；因此不超出保存资产的证据边界。','', '数据：`relations/`无GT图与帧、`evaluation/`局部/ROI后置关系及150 query参照、`analysis/`分层场景bootstrap与邻物负例、`supplement/opportunity_evidence.json`具体机会/误过滤证据。封存、合同、核验及代码快照保留；只本机保存，不GitHub发布，不修改旧阅读入口。']
    (OUT/'report.md').write_text('\n'.join(rows)+'\n')
    sources=[Path(__file__).resolve(),Path(__file__).with_name('evaluate_fi1_query_relation_diagnostic.py'),Path(__file__).with_name('verify_fi1_query_relation_diagnostic.py'),Path(__file__).with_name('supplement_fi1_query_relation_supply.py')]
    for p in sources:q=OUT/'code_snapshot'/p.relative_to(run.ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    seal(OUT/'completion_verified.json',dict(complete=True,report_sha256=sha(OUT/'report.md'),verification_sha256=sha(OUT/'verification.json'),supply_verification_sha256=sha(OUT/'supply_verification.json'),code={str(p):sha(p) for p in sources},decision='Local relational signal and candidate supply present; safe GT-free value-preserving grouping not established. No automatic tree/network/grouping rule.',training=False,production_modified=False,github_push=False));run.stage('complete',scenes=52,parents=207)
if __name__=='__main__':main()

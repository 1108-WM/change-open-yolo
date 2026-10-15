#!/usr/bin/env python3
"""Final evidence summary; preserves all earlier artifacts and corrections."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_nt_instance_ownership as run
D=run.D;OUT=run.OUT


def main():
    c=D.read(OUT/'contract.json');D.verify(c['code']);D.verify(D.read(OUT/'delivery_verified.json')['outputs'])
    s=D.read(OUT/'summary.json');cor=D.read(OUT/'failure_diagnosis_v2/summary.json');r=s['modes']['relation_C'];u=r['unions']
    case=next(b for b in D.read(OUT/'evaluation/scene0078_00/blocks.json') if b['candidate_id']=='c004' and b['region']==11)
    cc=[b for b in D.read(OUT/'evaluation/scene0078_00/blocks.json') if b['candidate_id']=='c004' and b['state']=='accept']
    target=sum(b['clipped_GT']['target'] for b in cc)
    assert target==342 and case['clipped_GT']['target']==177
    errata=dict(previous_report=str(OUT/'failure_diagnosis_v2/report.md'),
        reason='The scene0078/c004 accepted 9 blocks contain342 target points, not341 as stated in the supplemental prose. All sealed evaluation rows already report the correct counts. No rule, prediction or evaluation changes.',
        scene='scene0078_00',candidate_id='c004',correct_target_points=target,
        prediction_changes=False,evaluation_changes=False)
    D.seal(OUT/'final_report_errata.json',errata)
    baseline=s['modes']['all_C'];bu=baseline['unions']
    text=f'''# N/T实例关系归属原型：最终结果

2026-10-15。已实现并执行真实“接受／拒绝／无法确定”原型，开发验收未通过。范围为52个已暴露物理场景、207个采样T父、原1,308块；无独立验收或AP结论。

**N/T关系证据显著减少误补，但本轮固定规则把完整NT新增价值也过滤掉了。停止当前原型，不据此否定所有N/T关系方法或可学习性。**

## 固定方法

GT前封存：[contract.json](contract.json)及code_snapshot/。使用原N/T全点集的交集和双向覆盖，按固定几何key顺序做complete-link重复分组：组内任意两点集都互相覆盖≥一半；不通过大N的传递连接合并两个不重叠候选。

组内需存在不同几何的N与T、各至少100点，才作为暂定实例参照；同点集的N/T别名不重复计证据。某组任一成员覆盖父/块≥一半时提出归属。父和块各有唯一参照：同组接受、异组拒绝；缺参照或多组竞争无法确定。上述阈值均在GT前固定，无后置扫描。

query不参与归属；只对接受块复用前轮完全相同的query≥0.5受限新增点裁剪。每T联合全部接收块、最多一个副本，原父及完整N/T保留。已有query裁剪仅是待验证工具，尤其不忽略混合块会丢目标点的局限。

保存3,642个N/T节点、3,059个候选组（545个跨源参照）、全交集矩阵、每父/块的原候选覆盖及三态判断。覆盖原halo或原≤.15m邻居的候选均保存为局部节点，无前4邻居截断；原父、分块、halo、query集合和观测不变。几何关系假设不等于GT实例，N也不是正确答案。

## 全量固定union对照

| 指标 | 前轮全部块query裁剪 | N/T归属后同样裁剪 |
|---|---:|---:|
| 有动作父数 | {bu['actions']} | {u['actions']} |
| 完整NT同父新增目标（IoU增量≥.02） | {baseline['same_parent_pool']['pool_gain02']} | {r['same_parent_pool']['pool_gain02']} |
| 安全gain物理场景 | {baseline['safe_gain02_scenes']} | {r['safe_gain02_scenes']} |
| 稳定父新增目标精度（未知计入分母） | {baseline['stable_add_precision']:.2%} | {r['stable_add_precision']:.2%} |
| 好T副本IoU损失>.02 | {bu['good_loss_gt_02']}/12 | {u['good_loss_gt_02']}/12 |
| 已知邻物误补点（稳定父） | {bu['added_other']:,} | {u['added_other']:,} |
| 未知标注新增点（稳定父） | {bu['unknown_annotation_edits']:,} | {u['unknown_annotation_edits']:,} |
| 非稳定父动作 | {baseline['unresolved_actions']} | {r['unresolved_actions']} |

“好T副本损害”不是原T被覆盖，也不是已测得AP损害；原T仍在。安全gain是后置评估：同父相对完整NT增量≥.02且新增目标占比≥.9，不能用GT将其他副本事后剔除。稳定父精度只在143个稳定父计算；另64父保持目标/邻物null，分别报告含糊、超评估类别范围、未标注，不当作零错误。

新原型接受187块、拒绝95块、无法确定1026块。其中稳定父接收150块，裁剪输出10,019目标、353邻物、73未知；37个接收块属于非稳定父，亦未丢弃。稳定父55个动作、非稳定父11个动作。无任何原父点删除；仍有2个稳定父副本下降≥.02，好T最大损失为{u['worst_good_loss']:.6f}（低于.02）。连续同父池增量总和仅{r['same_parent_pool']['pool_delta_sum']:.9f}，没有预定gain02目标或阈值跨越。

## 为什么未能保住机会

前轮GT后点级机会贡献32块中，本轮接受11、无法确定21；16块缺父参照、5块缺块参照，未把它们强迫归给其他实例。全部unknown中1009块缺参照、17块多组竞争。只有11/32机会块存在原N同时多数覆盖父与块，恰好均被接收；这不是现有全部关系信息不可分的证明，只是本轮“多数覆盖＋跨源资格”证书的覆盖限制。

应区分两个GT后解释集合：32个点级机会块里有27纯目标块，而纯整块联合oracle有池价值的贡献块只有21个。后者本轮接收9、unknown12。两种集合不能混报；所有GT集合只解释固定输出，不参与推理。

scene0078/c004接收9个纯目标块，实际保留{target}目标点、无邻物；缺少块参照的r11遗漏177目标点，最终联合输出仍比完整NT差.020216。scene0002/r1因缺块参照unknown；scene0085、scene0301两个baseline收益父、scene0369/c006、scene0688等机会多因缺父参照停住。逐父连续增量与三态细节在[evaluation/opportunities.json](evaluation/opportunities.json)，覆盖全部点级机会父，并未丢掉其余负例。

55个稳定父动作中，52个修补副本不如或等于已有最佳N，且不如或等于实际连接N：当前规则更容易解释N已经覆盖的部分，归属一致未转化成完整NT新增价值。

N/T一致仍会错。接受集合中8个稳定父块裁剪后带邻物；例scene0199/c006通过连接N补入209个全邻物点，副本IoU下降.029332。这个N含5,115目标点与496邻物（仅8.84%污染）；即便N整体多数正确，其小污染仍可给错误局部块提供强连接。仅防巨大混合N或要求整体大致正确不足以保证块归属。55个父/N连接中17个含多个已知实例、1个已知邻物比例>.1；未知仍单列。

## 判定与边界

预先开发验收未通过：未保留≥2物理场景安全完整NT增量；且仍对身份未定父产生动作。后者是本轮预先声明的保守安全口径，即使放宽它，零gain的必要条件也仍失败。没有事后通过GT筛父来声称成功，也没有因减少动作/平均变好而继续。

停止本轮complete-link／跨源资格／唯一参照构造，不继续同轮调门槛、不扩大原修复器、不直接搭完整树、不进入独立场景/AP。失败不证明所有query/N/T关系方法无效，也不证明AP不可能受益。下一优先级按用户路线是有针对性的观测或原T逐帧关联来源：先在全部争议块及负例上固定检查协议，区分缺帧、遮挡、自动mask碎片化与真实冲突，再决定是否添加能共同看清父与块的帧。当前未自动新增观测。

只有明确规则先通过开发风险与池价值验收，才另封存未参与设计的物理隔离开发场景，排除已暴露52及变体、不动holdout；验证通过后再接自身语义和冻结SL/AP。

## 交付与审计

4项针对性测试通过；独立Python集合核验179,460个N/T对、3,059组、1,308块三态判断、414个联合输出、2,314个池目标/模式通过。baseline全部关键点数、损害、池结果与上一轮复现一致。生成进程禁止GT/旧evaluation读取；全部输出封存后才GT。代码与执行入口见[原始主报告](report.md)和[verification.json](verification.json)。

本文件是最终有效汇总。此前失败定位文字中两处汇总笔误已单独封存勘误：21/27块集合混称，以及scene0078接收点数341应为342；逐块评估数据始终正确，规则、预测、GT评估、原seal和停止判定均未改。见[集合口径勘误](failure_diagnosis_v2/summary.json)及[点数勘误](final_report_errata.json)。

未训练、未新增观测、未修改生产或项目阅读入口、未提交GitHub。全部结果仅限本轮采样父和原可编辑域。
'''
    path=OUT/'report_final.md';assert not path.exists();path.write_text(text)
    paths=[path,OUT/'final_report_errata.json',OUT/'summary.json',OUT/'decision.json',OUT/'verification.json',OUT/'contract.json',OUT/'completion_verified.json',OUT/'delivery_verified.json',Path(__file__)]
    for f in ('predictions_frozen.json','evaluation_frozen.json'):
        D.verify(D.read(OUT/f)['outputs']);paths.append(OUT/f)
    D.seal(OUT/'final_delivery_verified.json',dict(complete=True,current_report=str(path),outputs={str(p):D.sha(p) for p in paths},
        tests='4 passed',original_seals_unchanged=True,training=False,independent_validation_run=False,AP_run=False,github_push=False))
    print('final report and all seals verified',flush=True)


if __name__=='__main__':main()

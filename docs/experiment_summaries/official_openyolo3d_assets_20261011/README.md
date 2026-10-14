# Open-YOLO 3D 官方 Replica 候选掩码恢复

官方资产包链接：https://drive.google.com/file/d/1FneLaYaClWDO51L9lIvlTQbheh5SfOFD/view

来源：官方仓库 `scripts/get_class_agn_masks.sh`，对应上游commit `24f344d001d28d1cdd4087d3769c0d53c34d64cd`。包名 `OpenYOLO3D.zip`，与官方数据和权重下载脚本使用同一文件ID。

压缩包保存到 `download/OpenYOLO3D.zip`；只提取作者原生候选 `output/replica/replica_masks/{office0..4,room0..2}.pt` 到 `extracted/replica_masks/`，不提取真值掩码，也不覆盖项目既有权重／数据／结果。

下载和核验入口：`tools/verify_official_replica_mask_download.py`。完成时 `verified.json` 记录ZIP及8文件SHA256、逐文件CRC、点域／格式／数量。仅恢复资产，没有新预测、训练或评测；点数一致本身不是点序的独立证明，后续复现仍须核查官方数据身份。

已完成：8个文件格式／CRC／SHA256及完整点域通过；`mesh_identity_verified.json`确认包内8个原网格与本地现有网格逐字节一致，补足点序身份核验。压缩包约1.85GB，提取掩码约654MiB。2026-10-11已完成[官方掩码重评测](../replica_official_masks_full_20261011_v1/report.md)，按论文一位小数复现基线23.7／28.6／34.8；Full类别AP提高、实际去类AP下降。

2026-10-12已将8文件独立复制到标准目录 `output/replica/replica_masks/`，全部SHA256一致。[标准资产与复现说明](../../../output/replica/README.md)／[安装清单](../../../output/replica/official_replica_masks_manifest.json)。本目录源文件保留，旧合同与回执不改；本次安装没有新预测或评测。

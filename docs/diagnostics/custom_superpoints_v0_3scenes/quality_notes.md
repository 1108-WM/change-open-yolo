# Custom Superpoints v0 3-Scene Quality Notes

Scope: quality diagnostics only. No AP, no fusion main-flow changes, no SAM/YOLO/CLIP features.

Method source:
- Reused existing `tools/generate_geometric_superpoints.py` output under `output/geometric_superpoints_scannet200_even48_k025`.
- The generator is a Felzenszwalb-style kNN graph segmentation using xyz, normals, and color.
- Repository search did not find a standalone SAI3D/Open3DIS/MV3DIS/SAM-graph superpoint module.
- `models/Mask3D/third_party/ScanNet/Segmentator` is present as a mature ScanNet Felzenszwalb-Huttenlocher mesh-normal segmentator, but the source ScanNet PLY/data mount was unavailable in this session; therefore this v0 compares existing generated geometric superpoints to saved ScanNet200 built-in statistics.

Scene comparison:

| scene | built-in segments | custom segments | built-in median | custom median | built-in max | custom max | label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scene0011_00 | 1611 | 2974 | 72.0 | 55.0 | 31741 | 1068 | finer_and_less_overgrown |
| scene0077_00 | 683 | 1139 | 74.0 | 59.0 | 5037 | 702 | finer_and_less_overgrown |
| scene0608_01 | 1509 | 2483 | 70.0 | 54.0 | 9355 | 891 | finer_and_less_overgrown |

Initial readout:
- Custom geometric superpoints are consistently finer than the ScanNet200 built-in superpoints.
- The largest segments are much smaller in all three scenes, reducing the most obvious wall/floor/table overgrowth risk.
- `scene0011_00` still has one custom segment over 1000 points; the top-largest PNG should be inspected before using this as a replacement source.
- Several large custom segments have mixed semantic/instance purity, so this is not yet a drop-in replacement for final fusion.

Generated review files:
- `scene_summary.csv`: scene-level built-in vs custom statistics.
- `largest_custom_superpoints.csv`: largest custom segments for boundary/overgrowth review.
- `custom_superpoints.csv`: xyz/color/normal/bbox/planarity/adjacency-degree metadata.
- `custom_superpoint_adjacency.csv`: geometry/color/normal adjacency contacts.
- `custom_superpoint_purity.csv`: semantic/instance majority-purity proxy diagnostics.
- `../visual_checks/custom_superpoints_v0_3scenes/`: XZ scatter plots and purity histograms.

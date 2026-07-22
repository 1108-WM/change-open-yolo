# Superpoint Method Quality Comparison

Scope: diagnostic only. No AP, no fusion main-flow change, no GT-derived inference input.

Compared methods:
- `knn_geometry`: `output/geometric_superpoints_ibsp_v1_geometry_smoke`
- `mesh_normal`: `output/mesh_normal_python_default_3scenes`
- `mesh_normal_ibsp`: `output/mesh_normal_ibsp_sam_k070_3scenes`

Scene summary:

| scene | method | segments | median | max | sem weighted purity | inst weighted purity | sem mixed seg ratio | inst mixed seg ratio | pruned edges |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scene0011_00` | `knn_geometry` | 2974 | 55.0 | 1068 | 0.8425 | 0.9419 | 0.2898 | 0.1900 | 0 |
| `scene0011_00` | `mesh_normal` | 1620 | 72.0 | 31740 | 0.8716 | 0.9998 | 0.1969 | 0.0006 | 0 |
| `scene0011_00` | `mesh_normal_ibsp` | 1627 | 71.0 | 31740 | 0.8716 | 0.9998 | 0.2004 | 0.0006 | 324 |
| `scene0077_00` | `knn_geometry` | 1139 | 59.0 | 702 | 0.8412 | 0.9660 | 0.2687 | 0.1528 | 0 |
| `scene0077_00` | `mesh_normal` | 682 | 74.0 | 5037 | 0.8663 | 0.9999 | 0.1833 | 0.0015 | 0 |
| `scene0077_00` | `mesh_normal_ibsp` | 682 | 74.0 | 5037 | 0.8663 | 0.9999 | 0.1833 | 0.0015 | 22 |
| `scene0608_01` | `knn_geometry` | 2483 | 54.0 | 891 | 0.7409 | 0.9535 | 0.3733 | 0.1724 | 0 |
| `scene0608_01` | `mesh_normal` | 1507 | 70.0 | 9355 | 0.7593 | 0.9999 | 0.3119 | 0.0013 | 0 |
| `scene0608_01` | `mesh_normal_ibsp` | 1510 | 70.0 | 9355 | 0.7593 | 0.9998 | 0.3113 | 0.0013 | 608 |

Partition change relative to the first method:

| scene | reference | compared | split ref segments | split ref points | merged compared segments | merged compared points |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `scene0011_00` | `knn_geometry` | `mesh_normal` | 2211 | 195411 | 1496 | 232366 |
| `scene0011_00` | `knn_geometry` | `mesh_normal_ibsp` | 2212 | 195478 | 1496 | 232360 |
| `scene0077_00` | `knn_geometry` | `mesh_normal` | 881 | 78892 | 624 | 90193 |
| `scene0077_00` | `knn_geometry` | `mesh_normal_ibsp` | 881 | 78892 | 624 | 90193 |
| `scene0608_01` | `knn_geometry` | `mesh_normal` | 1939 | 164137 | 1376 | 185669 |
| `scene0608_01` | `knn_geometry` | `mesh_normal_ibsp` | 1940 | 164176 | 1377 | 185661 |

Readout:
- `mesh_normal` closely matches ScanNet Segmentator-style built-in superpoints in segment count and largest segment size; it is a mature geometry baseline rather than a finer replacement.
- The existing `mesh_normal_ibsp` run prunes very few graph edges, so its quality proxy stays almost identical to `mesh_normal` on these three scenes.
- The old kNN geometry run creates many more, smaller superpoints and removes the huge built-in regions, but its mixed semantic/instance segment ratios remain non-trivial.
- This comparison still uses GT semantic/instance columns only as offline purity proxies; it should not be interpreted as an AP result or an argument to connect IBSp to the main fusion path.

Generated files:
- `scene_method_summary.csv`
- `segment_purity.csv`
- `largest_segments.csv`
- `partition_overlap_vs_first_method.csv`
- `../visual_checks/superpoint_method_quality_3scenes/`

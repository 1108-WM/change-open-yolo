# Official Replica masks replay protocol

All eight scenes, original48 vocabulary and raw full mesh domains; author masks from `official_openyolo3d_assets_20261011/extracted/replica_masks` are shared by baseline and production Full. The downloaded author archive's eight original meshes are byte-identical to local meshes. No Mask3D inference, proposal thresholding or NMS is rerun.

Baseline executes unmodified upstream OpenYolo3D label/visibility/IoU methods against the frozen frequency1/.4/top40 Replica configuration and prior YOLO detections, with original uniform evaluation scores. Native confidence is a separate control. The local world projection implementation retains memory-safe chunks and verified original vertices; image/frame/pose/depth identity is bound before prediction.

Full retains registered GD10 + historical official100 D-v3 seven models + matched597 B26, NTUUprime/K800. It does not use experimental S/C/SL. The native branch labels official masks using the registered frequency10/.05/top600 configuration. Native-independent SAM/GD observations, superpoints and YOLO-SAM observations are reused by file hash. Native-dependent relations, filtering, U/Uprime, own evidence, full26D contexts and B26 output are recomputed.

The frozen prior own-semantics algorithm is used with its48-class adapter; no training, thresholds, model weights or class-specific rules are changed. Both methods' all-scene prediction receipts must be sealed before raw GT access. Semantic and actual-selected class-agnostic AP use the original evaluator; deduplication uses exact selected point sets and maximum selected score only, without refill or full-pool scoring.

Previous local-regenerated outputs remain unchanged. This experiment changes the original proposal asset; its Full metrics measure frozen cross-dataset transfer, not a new Replica-trained model. Seeing these results makes Replica exposed evaluation data.

Execution note: the initial score call used the D-v3 sklearn1.3.0 environment and failed on B26 tree-node dtype before saving scores. Its log and execution receipt are retained. `tools/resume_replica_official_scoring.py` resumed with B26 sklearn1.3.2 (same as previous inference/training), reusing completed sealed geometry/evidence; no model changes or training.

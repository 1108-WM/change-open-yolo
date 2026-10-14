"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

from tools.audit_train_candidate_quality_dataset import TRACK_SOURCE

def augment_union_features(
    rows: list[dict], union_lookup: dict, union_names: list[str],
) -> list[dict]:
    output = []
    seen_tracks = set()
    for row in rows:
        features = dict(row["model_features"])
        source_track = row["candidate_source"] == TRACK_SOURCE
        if source_track:
            key = (
                str(row["scene_name"]), int(row["relation_component_id"]),
                int(row["candidate_id"]),
            )
            if key not in union_lookup:
                raise ValueError(f"missing component union features: {key}")
            seen_tracks.add(key)
            for name in union_names:
                features[f"union__{name}"] = float(union_lookup[key][name])
            features["union__not_applicable"] = 0.0
        else:
            for name in union_names:
                features[f"union__{name}"] = 0.0
            features["union__not_applicable"] = 1.0
        output.append({**row, "model_features": features})
    if seen_tracks != set(union_lookup):
        missing = sorted(set(union_lookup) - seen_tracks)[:5]
        extra = sorted(seen_tracks - set(union_lookup))[:5]
        raise ValueError(f"union track coverage mismatch; missing={missing}, extra={extra}")
    return output

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_groundingdino_f30_boxes.py"
    spec = importlib.util.spec_from_file_location("export_groundingdino_f30_boxes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    def __call__(self, caption, return_offsets_mapping=False, **_):
        words = caption.replace(".", " .").split()
        ids = [101] + list(range(1, len(words) + 1)) + [102]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            offsets, cursor = [(0, 0)], 0
            for word in words:
                start = caption.index(word, cursor)
                offsets.append((start, start + len(word)))
                cursor = start + len(word)
            offsets.append((0, 0))
            result["offset_mapping"] = offsets
        return result


def test_prompt_groups_are_deterministic_and_below_text_limit():
    module = _module()
    groups = module._build_prompt_groups(["chair", "dining table", "door"], _Tokenizer(), 7)
    assert groups == [[0, 1], [2]]


def test_class_token_indices_keep_armchair_distinct_from_chair():
    module = _module()
    caption, _, spans = module._class_token_indices(_Tokenizer(), ["chair", "armchair"], [0, 1])
    assert caption == "chair. armchair."
    assert spans[0] != spans[1]


def test_sampling_reuses_fixed_f30_stride(tmp_path):
    module = _module()
    color = tmp_path / "color"
    color.mkdir()
    for index in range(30):
        (color / f"{index}.jpg").touch()
    paths = module._sample_color_paths(tmp_path, 1, 10, 3)
    assert [path.stem for path in paths] == ["0", "10", "20"]

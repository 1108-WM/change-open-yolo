import io
import struct

import numpy as np

from tools.prepare_scannet200_train_scene import _read_matrix, validate_split


def test_read_matrix_uses_little_endian_float32():
    expected = np.arange(16, dtype="<f4").reshape(4, 4)
    actual = _read_matrix(io.BytesIO(expected.tobytes()))
    np.testing.assert_array_equal(actual, expected)


def test_validate_split_accepts_train_only_scene(tmp_path):
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text("scene0001_01\n")
    validation.write_text("scene0011_00\n")
    validate_split("scene0001_01", train, [validation])


def test_validate_split_rejects_evaluation_overlap(tmp_path):
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text("scene0001_01\n")
    validation.write_text("scene0001_01\n")
    try:
        validate_split("scene0001_01", train, [validation])
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("expected overlap to be rejected")

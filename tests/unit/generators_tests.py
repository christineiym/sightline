"""
Unit tests for sightline/generators.py

Run with: pytest tests/unit/generators_tests.py

AI use disclosure: This was written with assistance from ChatGPT (GPT-5.3).
"""

import os
import csv
import pytest
from typing import List

from sightline.generators import Generator, FIELD_DEFAULT, FIELD_SYSTEMATIC, FIELD_RANDOM, SCOPE_IMAGE, SCOPE_OBJECT


class StubCharacteristic:
    def __init__(self, name, default=None, scope=SCOPE_IMAGE,
                 systematic_values=None, random_values=None):
        self.name = name
        self.default = default
        self.scope = scope
        self._systematic_values = systematic_values or []
        self._random_values = random_values or []

    def get_systematic_sample(self, n_max=None, step=None):
        return self._systematic_values

    def get_random_sample(self, n):
        # return list of length n
        if self._random_values:
            return self._random_values[:n]
        return [f"{self.name}_{i}" for i in range(n)]


class DummyGenerator(Generator):
    def __init__(self):
        self.generated_images = []

    def generate_contents(self, pair, constants, random_characteristics):
        # Return one "object" row per call
        row = {**constants, "object_id": 0}
        return [row], []

    def generate_image(self, out_dir: str, pair: bool, object_info, **kwargs):
        self.generated_images.append((out_dir, pair, object_info, kwargs))


class TestGenerateImageMetadata:
    """
    Tests for Generator.generate_image_metadata, covering:
    - default propagation
    - systematic cross product
    - repetition logic
    - random sampling integration
    """

    def _generator(self):
        return DummyGenerator()

    def _characteristics(self, default=None, systematic=None, random=None):
        return {
            FIELD_DEFAULT: default or [],
            FIELD_SYSTEMATIC: systematic or [],
            FIELD_RANDOM: random or [],
        }

    def test_defaults_are_applied(self):
        g = self._generator()
        char = self._characteristics(
            default=[StubCharacteristic("color", default="red")],
            systematic=[StubCharacteristic("shape", systematic_values=["circle"])],
            random=[]
        )

        result = g.generate_image_metadata(
            out_dir=".",
            pair=False,
            repetitions_per_combination=1,
            characteristics=char,
            step=[None],
            n=None
        )

        assert len(result) == 1
        assert result[0]["color"] == "red"
        assert result[0]["shape"] == "circle"

    def test_systematic_cross_product(self):
        g = self._generator()
        char = self._characteristics(
            systematic=[
                StubCharacteristic("shape", systematic_values=["circle", "square"]),
                StubCharacteristic("size", systematic_values=["small", "large"]),
            ]
        )

        result = g.generate_image_metadata(
            out_dir=".",
            pair=False,
            repetitions_per_combination=1,
            characteristics=char,
            step=[None, None],
            n=None
        )

        # 2 x 2 combinations
        assert len(result) == 4
        combos = {(r["shape"], r["size"]) for r in result}
        assert combos == {
            ("circle", "small"),
            ("circle", "large"),
            ("square", "small"),
            ("square", "large"),
        }

    def test_repetitions_per_combination(self):
        g = self._generator()
        char = self._characteristics(
            systematic=[StubCharacteristic("shape", systematic_values=["circle"])]
        )

        result = g.generate_image_metadata(
            out_dir=".",
            pair=False,
            repetitions_per_combination=3,
            characteristics=char,
            step=[None],
            n=None
        )

        assert len(result) == 3
        indices = [r["img_index"] for r in result]
        assert indices == [0, 1, 2]

    def test_random_values_image_scope(self):
        g = self._generator()
        char = self._characteristics(
            systematic=[StubCharacteristic("shape", systematic_values=["circle"])],
            random=[StubCharacteristic("noise", random_values=["a", "b", "c"])]
        )

        result = g.generate_image_metadata(
            out_dir=".",
            pair=False,
            repetitions_per_combination=3,
            characteristics=char,
            step=[None],
            n=None
        )

        assert len(result) == 9
        combos = {(r["shape"], r["noise"]) for r in result}
        assert combos == {
            ("circle", "a"),
            ("circle", "a"),
            ("circle", "a"),
            ("circle", "b"),
            ("circle", "b"),
            ("circle", "b"),
            ("circle", "c"),
            ("circle", "c"),
            ("circle", "c"),
        }

    def test_base_filename_format(self):
        g = self._generator()
        char = self._characteristics(
            systematic=[StubCharacteristic("shape", systematic_values=["circle"])]
        )

        result = g.generate_image_metadata(
            out_dir=".",
            pair=False,
            repetitions_per_combination=1,
            characteristics=char,
            step=[None],
            n=None
        )

        assert result[0]["base_filename"] == "00000"
        assert result[0]["images"] == "00000_a.png|00000_b.png"


class TestGenerateDataset:
    """
    Tests for Generator.generate_dataset:
    - CSV writing
    - return value
    """

    def _generator(self):
        return DummyGenerator()

    def test_csv_written(self, tmp_path):
        g = self._generator()
        out_csv = tmp_path / "out.csv"

        char = {
            FIELD_DEFAULT: [],
            FIELD_SYSTEMATIC: [StubCharacteristic("shape", systematic_values=["circle"])],
            FIELD_RANDOM: []
        }

        result_path = g.generate_dataset(
            out_dir=str(tmp_path),
            out_csv=str(out_csv),
            pair=False,
            repetitions_per_combination=1,
            characteristics=char,
            step=[None],
            n=None
        )

        assert os.path.exists(result_path)

        with open(result_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["shape"] == "circle"


class TestGenerateImagesFromFile:
    """
    Tests for Generator.generate_images_from_file:
    - grouping by id_column
    - correct invocation of generate_image
    """

    def _generator(self):
        return DummyGenerator()

    def test_grouping_and_generation(self, tmp_path):
        g = self._generator()
        csv_path = tmp_path / "input.csv"

        rows = [
            {"img_index": "0", "value": "a"},
            {"img_index": "0", "value": "b"},
            {"img_index": "1", "value": "c"},
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["img_index", "value"])
            writer.writeheader()
            writer.writerows(rows)

        grouped = g.generate_images_from_file(
            input_csv=str(csv_path),
            out_dir=str(tmp_path),
            pair=False,
            id_column="img_index"
        )

        assert len(grouped) == 2
        assert len(grouped["0"]) == 2
        assert len(grouped["1"]) == 1

        # Ensure generate_image was called once per group
        assert len(g.generated_images) == 2


class TestGenerateImageMetadataEdgeCases:
    """
    Edge case tests for Generator.generate_image_metadata:
    - empty characteristics
    - missing keys
    - mismatched step / n lengths
    - zero repetitions
    """

    def _generator(self):
        return DummyGenerator()

    def _characteristics(self, default=None, systematic=None, random=None):
        return {
            FIELD_DEFAULT: default or [],
            FIELD_SYSTEMATIC: systematic or [],
            FIELD_RANDOM: random or [],
        }

    def test_empty_all_characteristics(self):
        g = self._generator()
        char = self._characteristics()

        result = g.generate_image_metadata(
            out_dir=".",
            pair=False,
            repetitions_per_combination=1,
            characteristics=char,
            step=[],
            n=None
        )

        # No systematic values → no combinations → filler row with image metadata
        assert len(result) == 1
        assert "img_index" in result[0].keys()
        assert "base_filename" in result[0].keys()
        assert "images" in result[0].keys()

    def test_missing_systematic_key_raises(self):
        g = self._generator()
        char = {
            FIELD_DEFAULT: [],
            # FIELD_SYSTEMATIC missing
            FIELD_RANDOM: []
        }

        with pytest.raises(KeyError):
            g.generate_image_metadata(
                out_dir=".",
                pair=False,
                repetitions_per_combination=1,
                characteristics=char,
                step=[],
                n=None
            )

    def test_missing_random_key_raises(self):
        g = self._generator()
        char = {
            FIELD_DEFAULT: [],
            FIELD_SYSTEMATIC: [],
            # FIELD_RANDOM missing
        }

        with pytest.raises(KeyError):
            g.generate_image_metadata(
                out_dir=".",
                pair=False,
                repetitions_per_combination=1,
                characteristics=char,
                step=[],
                n=None
            )

    def test_step_shorter_than_systematic_raises(self):
        g = self._generator()
        char = self._characteristics(
            systematic=[
                StubCharacteristic("a", systematic_values=[1]),
                StubCharacteristic("b", systematic_values=[2]),
            ]
        )

        with pytest.raises(IndexError):
            g.generate_image_metadata(
                out_dir=".",
                pair=False,
                repetitions_per_combination=1,
                characteristics=char,
                step=[None],  # too short
                n=None
            )

    def test_n_shorter_than_systematic_raises(self):
        g = self._generator()
        char = self._characteristics(
            systematic=[
                StubCharacteristic("a", systematic_values=[1]),
                StubCharacteristic("b", systematic_values=[2]),
            ]
        )

        with pytest.raises(IndexError):
            g.generate_image_metadata(
                out_dir=".",
                pair=False,
                repetitions_per_combination=1,
                characteristics=char,
                step=[None, None],
                n=[1]  # too short
            )

    def test_zero_repetitions(self):
        g = self._generator()
        char = self._characteristics(
            systematic=[StubCharacteristic("shape", systematic_values=["circle"])]
        )

        result = g.generate_image_metadata(
            out_dir=".",
            pair=False,
            repetitions_per_combination=0,
            characteristics=char,
            step=[None],
            n=None
        )

        # No repetitions → no rows
        assert result == []

    def test_random_sample_shorter_than_expected(self):
        g = self._generator()
        char = self._characteristics(
            systematic=[StubCharacteristic("shape", systematic_values=["circle"])],
            random=[StubCharacteristic("noise", random_values=["only_one"])]
        )

        result = g.generate_image_metadata(
            out_dir=".",
            pair=False,
            repetitions_per_combination=3,
            characteristics=char,
            step=[None],
            n=None
        )

        # Behavior: generator does not guard length mismatch;
        # ensure at least first value propagates
        assert len(result) == 3
        assert result[0]["noise"] == "only_one"

    def test_object_scope_random_not_in_constants(self):
        g = self._generator()
        char = self._characteristics(
            systematic=[StubCharacteristic("shape", systematic_values=["circle"])],
            random=[StubCharacteristic("obj_noise", scope=SCOPE_OBJECT)]
        )

        result = g.generate_image_metadata(
            out_dir=".",
            pair=False,
            repetitions_per_combination=1,
            characteristics=char,
            step=[None],
            n=None
        )

        # Object-scoped randoms should not appear in top-level row
        assert "obj_noise" not in result[0]
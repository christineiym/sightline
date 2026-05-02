"""
Unit tests for sightline/characteristics.py

Run with: pytest tests/unit/characteristics_tests.py

AI use disclosure: This was written with assistance from Claude Sonnet 4.6.
"""
import pytest
from unittest.mock import MagicMock
import sightline.characteristics as characteristics_mod


# ---------------------------------------------------------------------------
# Fake domain classes
# These stand in for the real sightline.domains classes so that tests have
# no dependency on domain logic and remain fast/isolated.
# ---------------------------------------------------------------------------

class FakeContinuousDomain:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    def random_sample(self, *a, **kw): return ["rand"]
    def systematic_sample(self, *a, **kw): return ["sys"]

class FakeDiscreteRangeDomain:
    def __init__(self, min, max, unit):
        self.min = min
        self.max = max
        self.unit = unit
    def random_sample(self, *a, **kw): return ["rand"]
    def systematic_sample(self, *a, **kw): return ["sys"]

class FakeCategoricalDomain:
    def __init__(self, values):
        self.values = list(values)
    def random_sample(self, *a, **kw): return ["rand"]
    def systematic_sample(self, *a, **kw): return ["sys"]


@pytest.fixture(autouse=True)
def patch_domains(monkeypatch):
    """
    Patch the three domain constructors at the point where characteristics.py
    looks them up — i.e. the names already bound in the sightline.characteristics
    module namespace after its 'from sightline.domains import ...' ran.
    """
    monkeypatch.setattr(characteristics_mod, "ContinuousDomain", FakeContinuousDomain)
    monkeypatch.setattr(characteristics_mod, "DiscreteRangeDomain", FakeDiscreteRangeDomain)
    monkeypatch.setattr(characteristics_mod, "CategoricalDomain", FakeCategoricalDomain)


@pytest.fixture()
def mod():
    return characteristics_mod


# ===========================================================================
# Characteristic (abstract base) — tested via ContinuousCharacteristic
# ===========================================================================

class TestCharacteristicBase:
    def test_name_property(self, mod):
        c = mod.ContinuousCharacteristic("brightness", 0, 100, 50, "global")
        assert c.name == "brightness"

    def test_default_property(self, mod):
        c = mod.ContinuousCharacteristic("contrast", 0, 255, 128, "local")
        assert c.default == 128

    def test_scope_property(self, mod):
        c = mod.ContinuousCharacteristic("gamma", 0.1, 10.0, 1.0, "per-image")
        assert c.scope == "per-image"

    def test_domain_property_returns_domain(self, mod):
        c = mod.ContinuousCharacteristic("x", 0, 1, 0.5, "s")
        # domain should be the fake ContinuousDomain instance
        assert hasattr(c.domain, "min") and hasattr(c.domain, "max")

    def test_get_random_sample_delegates_to_domain(self, mod):
        c = mod.ContinuousCharacteristic("x", 0, 1, 0.5, "s")
        result = c.get_random_sample(n=3, with_replacement=False)
        assert result == ["rand"]

    def test_get_random_sample_passes_args_through(self, mod):
        c = mod.ContinuousCharacteristic("x", 0, 1, 0.5, "s")
        # domain is a fake; just verify no TypeError is raised with various args
        c.get_random_sample()
        c.get_random_sample(5)
        c.get_random_sample(n=2, with_replacement=True)

    def test_get_systematic_sample_delegates_to_domain(self, mod):
        c = mod.ContinuousCharacteristic("x", 0, 1, 0.5, "s")
        result = c.get_systematic_sample()
        assert result == ["sys"]

    def test_characteristic_is_abstract(self, mod):
        """Characteristic cannot be instantiated directly."""
        with pytest.raises(TypeError):
            mod.Characteristic("x", MagicMock(), 0, "s")


# ===========================================================================
# ContinuousCharacteristic
# ===========================================================================

class TestContinuousCharacteristic:
    def test_construction(self, mod):
        c = mod.ContinuousCharacteristic("brightness", 0, 255, 128, "global")
        assert c.name == "brightness"
        assert c.default == 128
        assert c.scope == "global"

    def test_min_property(self, mod):
        c = mod.ContinuousCharacteristic("x", 10, 200, 100, "s")
        assert c.min == 10

    def test_max_property(self, mod):
        c = mod.ContinuousCharacteristic("x", 10, 200, 100, "s")
        assert c.max == 200

    def test_min_max_float(self, mod):
        c = mod.ContinuousCharacteristic("gamma", 0.1, 9.9, 1.0, "s")
        assert c.min == pytest.approx(0.1)
        assert c.max == pytest.approx(9.9)

    def test_min_equals_max(self, mod):
        """Edge case: degenerate single-value range."""
        c = mod.ContinuousCharacteristic("fixed", 5, 5, 5, "s")
        assert c.min == c.max == 5

    def test_default_at_boundary_min(self, mod):
        c = mod.ContinuousCharacteristic("x", 0, 100, 0, "s")
        assert c.default == 0

    def test_default_at_boundary_max(self, mod):
        c = mod.ContinuousCharacteristic("x", 0, 100, 100, "s")
        assert c.default == 100

    def test_domain_type(self, mod):
        c = mod.ContinuousCharacteristic("x", 0, 1, 0.5, "s")
        assert isinstance(c.domain, FakeContinuousDomain)


# ===========================================================================
# DiscreteRangeCharacteristic
# ===========================================================================

class TestDiscreteRangeCharacteristic:
    def test_construction(self, mod):
        c = mod.DiscreteRangeCharacteristic("iso", 100, 6400, 1, 400, "camera")
        assert c.name == "iso"
        assert c.default == 400
        assert c.scope == "camera"

    def test_min_property(self, mod):
        c = mod.DiscreteRangeCharacteristic("iso", 100, 6400, 1, 400, "camera")
        assert c.min == 100

    def test_max_property(self, mod):
        c = mod.DiscreteRangeCharacteristic("iso", 100, 6400, 1, 400, "camera")
        assert c.max == 6400
    
    def test_unit_property(self, mod):
        c = mod.DiscreteRangeCharacteristic("iso", 100, 6400, 1, 400, "camera")
        assert c.unit == 1

    def test_min_equals_max(self, mod):
        c = mod.DiscreteRangeCharacteristic("fixed", 50, 50, 1, 50, "s")
        assert c.min == c.max == 50

    def test_domain_type(self, mod):
        c = mod.DiscreteRangeCharacteristic("x", 0, 10, 1, 5, "s")
        assert isinstance(c.domain, FakeDiscreteRangeDomain)

    def test_random_sample_delegates(self, mod):
        c = mod.DiscreteRangeCharacteristic("x", 0, 10, 1, 5, "s")
        assert c.get_random_sample() == ["rand"]

    def test_systematic_sample_delegates(self, mod):
        c = mod.DiscreteRangeCharacteristic("x", 0, 10, 1, 5, "s")
        assert c.get_systematic_sample() == ["sys"]


# ===========================================================================
# CategoricalCharacteristic
# ===========================================================================

class TestCategoricalCharacteristic:
    def test_construction(self, mod):
        c = mod.CategoricalCharacteristic("format", ["jpeg", "png", "tiff"], "jpeg", "export")
        assert c.name == "format"
        assert c.default == "jpeg"
        assert c.scope == "export"

    def test_values_property(self, mod):
        vals = ["low", "medium", "high"]
        c = mod.CategoricalCharacteristic("quality", vals, "medium", "s")
        assert c.values == vals

    def test_values_returns_deep_copy(self, mod):
        """Mutating the returned list must not affect the stored domain values."""
        c = mod.CategoricalCharacteristic("color", ["red", "green", "blue"], "red", "s")
        returned = c.values
        returned.append("purple")
        assert "purple" not in c.values

    def test_values_with_non_string_types(self, mod):
        c = mod.CategoricalCharacteristic("level", [1, 2, 3], 2, "s")
        assert c.values == [1, 2, 3]

    def test_values_single_item(self, mod):
        c = mod.CategoricalCharacteristic("only", ["sole"], "sole", "s")
        assert c.values == ["sole"]

    def test_values_accepts_generator(self, mod):
        """Constructor should handle any Iterable, not just lists."""
        c = mod.CategoricalCharacteristic("gen", (x for x in range(3)), 0, "s")
        assert c.values == [0, 1, 2]

    def test_domain_type(self, mod):
        c = mod.CategoricalCharacteristic("x", ["a"], "a", "s")
        assert isinstance(c.domain, FakeCategoricalDomain)

    def test_random_sample_delegates(self, mod):
        c = mod.CategoricalCharacteristic("x", ["a", "b"], "a", "s")
        assert c.get_random_sample() == ["rand"]

    def test_systematic_sample_delegates(self, mod):
        c = mod.CategoricalCharacteristic("x", ["a", "b"], "a", "s")
        assert c.get_systematic_sample() == ["sys"]


# ===========================================================================
# Cross-subclass / polymorphism checks
# ===========================================================================

class TestPolymorphism:
    def test_all_subclasses_have_name(self, mod):
        chars = [
            mod.ContinuousCharacteristic("a", 0, 1, 0.5, "s"),
            mod.DiscreteRangeCharacteristic("b", 0, 10, 1, 5, "s"),
            mod.CategoricalCharacteristic("c", ["x"], "x", "s"),
        ]
        for ch in chars:
            assert isinstance(ch.name, str)

    def test_all_subclasses_have_domain(self, mod):
        chars = [
            mod.ContinuousCharacteristic("a", 0, 1, 0.5, "s"),
            mod.DiscreteRangeCharacteristic("b", 0, 10, 1, 5, "s"),
            mod.CategoricalCharacteristic("c", ["x"], "x", "s"),
        ]
        for ch in chars:
            assert ch.domain is not None

    def test_all_subclasses_support_random_sample(self, mod):
        chars = [
            mod.ContinuousCharacteristic("a", 0, 1, 0.5, "s"),
            mod.DiscreteRangeCharacteristic("b", 0, 10, 1, 5, "s"),
            mod.CategoricalCharacteristic("c", ["x"], "x", "s"),
        ]
        for ch in chars:
            result = ch.get_random_sample()
            assert result == ["rand"]

    def test_all_subclasses_support_systematic_sample(self, mod):
        chars = [
            mod.ContinuousCharacteristic("a", 0, 1, 0.5, "s"),
            mod.DiscreteRangeCharacteristic("b", 0, 10, 1, 5, "s"),
            mod.CategoricalCharacteristic("c", ["x"], "x", "s"),
        ]
        for ch in chars:
            result = ch.get_systematic_sample()
            assert result == ["sys"]
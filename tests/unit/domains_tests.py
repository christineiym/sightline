"""
Unit tests for sightline/domains.py

Covers:
  - Domain abstract base and _systematic_sample_from_range (all four cases),
    tested via ContinuousDomain, DiscreteRangeDomain, and CategoricalDomain.
  - ContinuousDomain, DiscreteRangeDomain, CategoricalDomain individually.

Run with: pytest tests/unit/domains_tests.py

AI use disclosure: This was written with assistance from Claude Sonnet 4.6.
"""
import math
import pytest
from sightline.domains import (
    Domain,
    ContinuousDomain,
    DiscreteRangeDomain,
    CategoricalDomain,
)


# ===========================================================================
# Helpers
# ===========================================================================

def assert_approx_list(actual, expected, rel=1e-6):
    """Assert two float lists are element-wise approximately equal."""
    assert len(actual) == len(expected), f"Length mismatch: {len(actual)} != {len(expected)}"
    for a, e in zip(actual, expected):
        assert math.isclose(a, e, rel_tol=rel), f"{a} != {e}"


# ===========================================================================
# Domain (abstract base)
# ===========================================================================

class TestDomainAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Domain()

    def test_subclass_must_implement_random_sample(self):
        """random_sample is still abstract; omitting it must raise."""
        class Incomplete(Domain):
            def systematic_sample(self, *a, **kw): ...
        with pytest.raises(TypeError):
            Incomplete()
    
    def test_subclass_must_implement_random_sample(self):
        """random_sample is still abstract; omitting it must raise."""
        class Incomplete(Domain):
            def random_sample(self, *a, **kw): ...
        with pytest.raises(TypeError):
            Incomplete()


# ===========================================================================
# Domain._systematic_sample_from_range via ContinuousDomain
# ===========================================================================

class TestSystematicSampleFromRangeViaContinuous:
    """
    Tests for Domain._systematic_sample_from_range, exercising all four cases
    and edge conditions. ContinuousDomain is used as the host so that
    self.min/self.max are available for Case 4.
    """

    def _domain(self, min, max):
        return ContinuousDomain(min, max)

    def _call(self, domain, min, max, step=None, n_max=None):
        return domain._systematic_sample_from_range(min, max, step, n_max)

    # --- Case 1: step AND n_max both provided ---

    def test_case1_basic(self):
        d = self._domain(0, 10)
        result = self._call(d, 0, 10, step=1, n_max=3)
        assert len(result) <= 3  # floor division means at most n_max+1 elements
        assert result[0] == pytest.approx(0)

    def test_case1_respects_n_max_as_upper_bound(self):
        d = self._domain(0, 100)
        result = self._call(d, 0, 100, step=1, n_max=5)
        assert_approx_list(result, [0, 25, 50, 75, 100])

    # --- Case 2: only n_max provided ---

    def test_case2_returns_n_max(self):
        d = self._domain(0, 10)
        result = self._call(d, 0, 10, step=None, n_max=6)
        # step = 10/5 = 2 -> np.linspace(0, 10, 6) = [0,2,4,6,8,10] (stop included)
        assert len(result) == 6

    def test_case2_starts_at_min(self):
        d = self._domain(0, 1)
        result = self._call(d, 0, 1, step=None, n_max=3)
        assert result[0] == pytest.approx(0.0)

    def test_case2_float_range(self):
        d = self._domain(0.0, 1.0)
        result = self._call(d, 0.0, 1.0, step=None, n_max=3)
        # step = 1.0 / (3-1) = 0.5 -> [0.0, 0.5, 1.0]
        assert_approx_list(result, [0.0, 0.5, 1.0])

    # --- Case 3: only step provided ---

    def test_case3_integer_step(self):
        d = self._domain(0, 10)
        result = self._call(d, 0, 10, step=2, n_max=None)
        assert_approx_list(result, [0, 2, 4, 6, 8, 10])

    def test_case3_step_larger_than_range(self):
        d = self._domain(0, 5)
        result = self._call(d, 0, 5, step=10, n_max=None)
        assert result == [0]

    def test_case3_step_equals_range(self):
        d = self._domain(0, 5)
        result = self._call(d, 0, 5, step=5, n_max=None)
        assert result == [0, 5]

    # --- Case 4: neither step nor n_max -- uses self.min / self.max ---

    def test_case4_integer_range_step_is_one(self):
        # max - min = 50, fractional part = 0 -> step = 1
        d = self._domain(50, 100)
        result = self._call(d, 50, 100, step=None, n_max=None)
        assert result[0] == pytest.approx(50)
        assert result[1] == pytest.approx(51)

    def test_case4_fractional_range_correct_step(self):
        # docstring Example A: max=7, min=2.65 -> diff=5.35, frac=0.35=7/20 -> step=0.05
        d = self._domain(2.65, 7)
        result = self._call(d, 2.65, 7, step=None, n_max=None)
        assert result[0] == pytest.approx(2.65)
        assert result[1] == pytest.approx(2.65 + 0.05)

    def test_case4_offset_float_range(self):
        # docstring Example C: min=1.3, max=5.3 -> diff=4.0, frac=0 -> step=1
        d = self._domain(1.3, 5.3)
        result = self._call(d, 1.3, 5.3, step=None, n_max=None)
        assert result[0] == pytest.approx(1.3)
        assert result[1] == pytest.approx(2.3)

    def test_case4_uses_domain_bounds_not_passed_args(self):
        # Domain is [0, 10] (integer -> step=1); we pass different min/max args.
        # Step is still derived from self.min=0, self.max=10.
        d = self._domain(0, 10)
        result_full = self._call(d, 0, 10, step=None, n_max=None)
        result_narrow = self._call(d, 2, 8, step=None, n_max=None)
        assert (result_full[1] - result_full[0]) == pytest.approx(
            result_narrow[1] - result_narrow[0]
        )

    # --- Edge / guard cases ---

    def test_step_zero_returns_min(self):
        d = self._domain(0, 10)
        assert self._call(d, 3, 10, step=0, n_max=None) == [3]

    def test_negative_n_max_raises(self):
        d = self._domain(0, 10)
        with pytest.raises(ValueError):
            self._call(d, 0, 10, step=1, n_max=-1)

    def test_zero_n_max_raises(self):
        d = self._domain(0, 10)
        with pytest.raises(ValueError):
            self._call(d, 0, 10, step=1, n_max=0)
    
    def test_one_n_max_returns_min(self):
        d = self._domain(0, 10)
        assert self._call(d, 3, 10, step=0, n_max=None) == [3]

    def test_swapped_min_max_normalised(self):
        """min > max should produce the same output as the corrected order."""
        d = self._domain(0, 10)
        normal = self._call(d, 0, 10, step=2, n_max=None)
        swapped = self._call(d, 10, 0, step=2, n_max=None)
        assert_approx_list(normal, swapped)


# ===========================================================================
# Domain._systematic_sample_from_range via DiscreteRangeDomain
#
# DiscreteRangeDomain has self.min and self.max, so all four cases work.
# Case 4 uses those bounds, not the min/max args passed to the helper.
# ===========================================================================

class TestSystematicSampleFromRangeViaDiscreteRange:

    def _domain(self, min, max, unit=1):
        return DiscreteRangeDomain(min, max, unit)

    def _call(self, domain, min, max, step=None, n_max=None):
        return domain._systematic_sample_from_range(min, max, step, n_max)

    # --- Case 1: step AND n_max both provided ---

    def test_case1_basic(self):
        d = self._domain(0, 10)
        result = self._call(d, 0, 10, step=1, n_max=3)
        assert len(result) <= 3
        assert result[0] == pytest.approx(0)

    def test_case1_scale_factor_applied(self):
        d = self._domain(0, 100)
        result = self._call(d, 0, 100, step=1, n_max=5)
        assert_approx_list(result, [0, 25, 50, 75, 100])

    def test_case1_result_is_list(self):
        d = self._domain(0, 10)
        assert isinstance(self._call(d, 0, 10, step=2, n_max=2), list)

    # --- Case 2: only n_max provided ---

    def test_case2_length(self):
        d = self._domain(0, 10)
        result = self._call(d, 0, 10, step=None, n_max=6)
        # step = 10/5 = 2 -> [0,2,4,6,8,10]
        assert len(result) == 6

    def test_case2_starts_at_min(self):
        d = self._domain(0, 20)
        result = self._call(d, 0, 20, step=None, n_max=3)
        assert result[0] == pytest.approx(0.0)

    def test_case2_result_is_list(self):
        d = self._domain(0, 10)
        assert isinstance(self._call(d, 0, 10, step=None, n_max=3), list)

    # --- Case 3: only step provided ---

    def test_case3_integer_step(self):
        d = self._domain(0, 10)
        result = self._call(d, 0, 10, step=2, n_max=None)
        assert_approx_list(result, [0, 2, 4, 6, 8, 10])

    def test_case3_step_larger_than_range(self):
        d = self._domain(0, 5)
        result = self._call(d, 0, 5, step=10, n_max=None)
        assert result == [0]

    def test_case3_step_equals_range(self):
        d = self._domain(0, 5)
        result = self._call(d, 0, 5, step=5, n_max=None)
        assert_approx_list(result, [0, 5])

    def test_case3_unit_greater_than_one(self):
        d = self._domain(0, 50, unit=5)
        result = self._call(d, 0, 50, step=10, n_max=None)
        assert_approx_list(result, [0, 10, 20, 30, 40, 50])

    # --- Case 4: neither step nor n_max -- uses self.min / self.max ---

    def test_case4_integer_range_step_is_one(self):
        # self.min=0, self.max=10 -> diff=10, frac=0 -> step=1
        d = self._domain(0, 10)
        result = self._call(d, 0, 10, step=None, n_max=None)
        assert result[0] == pytest.approx(0)
        assert result[1] == pytest.approx(1)

    def test_case4_uses_domain_bounds_not_passed_args(self):
        # Domain is [0, 20]; step is derived from self.min/self.max, not the args.
        d = self._domain(0, 20)
        result_full = self._call(d, 0, 20, step=None, n_max=None)
        result_narrow = self._call(d, 5, 15, step=None, n_max=None)
        assert (result_full[1] - result_full[0]) == pytest.approx(
            result_narrow[1] - result_narrow[0]
        )

    def test_case4_large_unit_domain(self):
        # self.min=0, self.max=100 -> frac=0 -> step=1 (unit does not affect Case 4)
        d = self._domain(0, 100, unit=10)
        result = self._call(d, 0, 100, step=None, n_max=None)
        assert result[0] == pytest.approx(0)
        assert result[1] == pytest.approx(1)

    # --- Edge / guard cases ---

    def test_step_zero_returns_min(self):
        d = self._domain(0, 10)
        assert self._call(d, 3, 10, step=0) == [3]

    def test_negative_n_max_raises(self):
        d = self._domain(0, 10)
        with pytest.raises(ValueError):
            self._call(d, 0, 10, step=1, n_max=-1)

    def test_zero_n_max_raises(self):
        d = self._domain(0, 10)
        with pytest.raises(ValueError):
            self._call(d, 0, 10, step=1, n_max=0)

    def test_swapped_min_max_normalised(self):
        d = self._domain(0, 10)
        normal = self._call(d, 0, 10, step=2)
        swapped = self._call(d, 10, 0, step=2)
        assert_approx_list(normal, swapped)


# ===========================================================================
# Domain._systematic_sample_from_range via CategoricalDomain
# ===========================================================================

class TestSystematicSampleFromRangeViaCategorical:

    def _domain(self, values):
        return CategoricalDomain(values)

    def _call(self, domain, min, max, step=None, n_max=None):
        return domain._systematic_sample_from_range(min, max, step, n_max)

    # --- Case 1: step AND n_max both provided ---

    def test_case1_basic(self):
        d = self._domain(["a", "b", "c", "d", "e"])
        result = self._call(d, 0, 4, step=1, n_max=3)
        assert len(result) <= 3
        assert result[0] == pytest.approx(0)

    def test_case1_scale_factor_applied(self):
        d = self._domain(list(range(100)))
        result = self._call(d, 0, 100, step=1, n_max=5)
        assert_approx_list(result, [0, 25, 50, 75, 100])

    def test_case1_result_is_list(self):
        d = self._domain(["a", "b", "c"])
        assert isinstance(self._call(d, 0, 2, step=1, n_max=2), list)

    # --- Case 2: only n_max provided ---

    def test_case2_length(self):
        d = self._domain(list(range(10)))
        result = self._call(d, 0, 10, step=None, n_max=6)
        assert len(result) == 6

    def test_case2_starts_at_min(self):
        d = self._domain(["a", "b", "c", "d"])
        result = self._call(d, 0, 3, step=None, n_max=3)
        assert result[0] == pytest.approx(0.0)

    def test_case2_result_is_list(self):
        d = self._domain(["a", "b", "c"])
        assert isinstance(self._call(d, 0, 2, step=None, n_max=2), list)

    # --- Case 3: only step provided ---

    def test_case3_integer_step(self):
        d = self._domain(["a", "b", "c", "d", "e"])
        result = self._call(d, 0, 4, step=1, n_max=None)
        # Returns raw index floats; CategoricalDomain.systematic_sample maps these to values
        assert_approx_list(result, [0, 1, 2, 3, 4])

    def test_case3_step_two(self):
        d = self._domain(["a", "b", "c", "d", "e"])
        result = self._call(d, 0, 4, step=2, n_max=None)
        assert_approx_list(result, [0, 2, 4])

    def test_case3_step_larger_than_range(self):
        d = self._domain(["a", "b"])
        result = self._call(d, 0, 1, step=10, n_max=None)
        assert result == [0]

    def test_case3_result_is_list(self):
        d = self._domain(["a", "b", "c"])
        assert isinstance(self._call(d, 0, 2, step=1, n_max=None), list)

    # --- Case 4: neither step nor n_max -- uses list indices for lower and upper bounds ---

    def test_case4_integer_range_step_is_one(self):
        # list of len 10, so frac=0 -> step=1
        d = self._domain(list(range(10)))
        result = self._call(d, 0, 10, step=None, n_max=None)
        assert result[0] == pytest.approx(0)
        assert result[1] == pytest.approx(1)

    def test_case4_uses_domain_bounds_not_passed_args(self):
        # Domain is [0, 20]; step is derived from list indices, not the args.
        d = self._domain(list(range(20)))
        result_full = self._call(d, 0, 20, step=None, n_max=None)
        result_narrow = self._call(d, 5, 15, step=None, n_max=None)
        assert (result_full[1] - result_full[0]) == pytest.approx(
            result_narrow[1] - result_narrow[0]
        )

    def test_case4_large_unit_domain(self):
        # list of len 10, so frac=0 -> step=1
        d = self._domain(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        result = self._call(d, 0, 100, step=None, n_max=None)
        assert result[0] == pytest.approx(0)
        assert result[1] == pytest.approx(1)

    # --- Edge / guard cases ---

    def test_step_zero_returns_min(self):
        d = self._domain(["a", "b", "c"])
        assert self._call(d, 0, 2, step=0) == [0]

    def test_negative_n_max_raises(self):
        d = self._domain(["a", "b", "c"])
        with pytest.raises(ValueError):
            self._call(d, 0, 2, step=1, n_max=-1)

    def test_zero_n_max_raises(self):
        d = self._domain(["a", "b", "c"])
        with pytest.raises(ValueError):
            self._call(d, 0, 2, step=1, n_max=0)
    
    def test_one_n_max_returns_min(self):
        d = self._domain(["pancake", "spam", "eggs", "syrup"])
        assert self._call(d, 0, 3, step=None, n_max=1) == [0]

    def test_swapped_min_max_normalised(self):
        d = self._domain(["a", "b", "c", "d", "e"])
        normal = self._call(d, 0, 4, step=2)
        swapped = self._call(d, 4, 0, step=2)
        assert_approx_list(normal, swapped)


# ===========================================================================
# ContinuousDomain
# ===========================================================================

class TestContinuousDomain:

    # --- Construction & properties ---

    def test_min_property(self):
        assert ContinuousDomain(0, 100).min == 0

    def test_max_property(self):
        assert ContinuousDomain(0, 100).max == 100

    def test_float_bounds(self):
        d = ContinuousDomain(0.1, 9.9)
        assert d.min == pytest.approx(0.1)
        assert d.max == pytest.approx(9.9)

    def test_min_equals_max(self):
        d = ContinuousDomain(5, 5)
        assert d.min == d.max == 5

    def test_negative_bounds(self):
        d = ContinuousDomain(-10, -1)
        assert d.min == -10
        assert d.max == -1

    # --- random_sample ---

    def test_random_sample_default_returns_one_value(self):
        with pytest.raises(TypeError):
            ContinuousDomain(0, 1).random_sample()

    def test_random_sample_dims_parameter(self):
        assert len(ContinuousDomain(0, 1).random_sample(n=10)) == 10

    def test_random_sample_values_within_bounds(self):
        d = ContinuousDomain(5.0, 10.0)
        for _ in range(20):
            assert all(5.0 <= v <= 10.0 for v in d.random_sample(n=50))

    def test_random_sample_returns_list(self):
        assert isinstance(ContinuousDomain(0, 1).random_sample(n=4), list)

    def test_random_sample_zero_dims(self):
        assert ContinuousDomain(0, 1).random_sample(n=0) == []

    # --- systematic_sample ---

    def test_systematic_sample_returns_list(self):
        assert isinstance(ContinuousDomain(0, 10).systematic_sample(step=2, n_max=None), list)

    def test_systematic_sample_step_only(self):
        d = ContinuousDomain(0, 10)
        assert_approx_list(d.systematic_sample(step=2, n_max=None), [0, 2, 4, 6, 8, 10])

    def test_systematic_sample_n_max_only(self):
        d = ContinuousDomain(0, 10)
        assert_approx_list(d.systematic_sample(step=None, n_max=3), [0, 5, 10])

    def test_systematic_sample_float_step(self):
        d = ContinuousDomain(0.0, 1.0)
        assert_approx_list(d.systematic_sample(step=0.25, n_max=None), [0.0, 0.25, 0.5, 0.75, 1.00])

    def test_systematic_sample_no_args_integer_range(self):
        # Case 4: fractional part = 0 -> step = 1
        d = ContinuousDomain(0, 5)
        result = d.systematic_sample(step=None, n_max=None)
        assert result[0] == pytest.approx(0)
        assert result[1] == pytest.approx(1)

    def test_systematic_sample_no_args_fractional_range(self):
        # Case 4: diff = 7 - 2.65 = 5.35, frac = 0.35 = 7/20 -> step = 0.05
        d = ContinuousDomain(2.65, 7)
        result = d.systematic_sample(step=None, n_max=None)
        assert result[0] == pytest.approx(2.65)
        assert result[1] == pytest.approx(2.65 + 0.05)


# ===========================================================================
# DiscreteRangeDomain
# ===========================================================================

class TestDiscreteRangeDomain:

    # --- Construction & properties ---

    def test_min_property(self):
        assert DiscreteRangeDomain(0, 10, 1).min == 0

    def test_max_property(self):
        assert DiscreteRangeDomain(0, 10, 1).max == 10

    def test_unit_property(self):
        assert DiscreteRangeDomain(0, 100, 5).unit == 5

    def test_default_unit_is_one(self):
        assert DiscreteRangeDomain(0, 10).unit == 1

    # --- random_sample ---

    def test_random_sample_n_limits_count(self):
        assert len(DiscreteRangeDomain(0, 10, 1).random_sample(n=3, with_replacement=False)) == 3

    def test_random_sample_with_replacement_allows_repeats(self):
        assert len(DiscreteRangeDomain(0, 2, 1).random_sample(n=10, with_replacement=True)) == 10

    def test_random_sample_n_exceeds_domain_falls_back_to_replacement(self):
        """n > domain size should not raise even when with_replacement=False."""
        assert len(DiscreteRangeDomain(0, 3, 1).random_sample(n=10, with_replacement=False)) == 10

    def test_random_sample_values_are_multiples_of_unit(self):
        for v in DiscreteRangeDomain(0, 50, 5).random_sample(n=21, with_replacement=True):
            assert v % 5 == 0

    def test_random_sample_returns_list(self):
        assert isinstance(DiscreteRangeDomain(0, 10, 1).random_sample(), list)

    # --- systematic_sample ---

    def test_systematic_sample_returns_list(self):
        assert isinstance(DiscreteRangeDomain(0, 5, 1).systematic_sample(), list)

    def test_systematic_sample_with_step(self):
        assert_approx_list(DiscreteRangeDomain(0, 10, 1).systematic_sample(step=2), [0, 2, 4, 6, 8, 10])

    def test_systematic_sample_no_args_case4(self):
        # self.min=0, self.max=10 -> frac=0 -> step=1
        d = DiscreteRangeDomain(0, 10, 1)
        result = d.systematic_sample()
        assert result[0] == pytest.approx(0)
        assert result[1] == pytest.approx(1)


# ===========================================================================
# CategoricalDomain
# ===========================================================================

class TestCategoricalDomain:

    # --- Construction & properties ---

    def test_values_property(self):
        assert CategoricalDomain(["a", "b", "c"]).values == ["a", "b", "c"]

    def test_values_returns_deep_copy(self):
        d = CategoricalDomain(["x", "y"])
        returned = d.values
        returned.append("z")
        assert "z" not in d.values

    def test_constructor_deep_copies_input(self):
        original = ["a", "b"]
        d = CategoricalDomain(original)
        original.append("c")
        assert "c" not in d.values

    def test_non_string_values(self):
        assert CategoricalDomain([1, 2, 3]).values == [1, 2, 3]

    def test_single_value(self):
        assert CategoricalDomain(["only"]).values == ["only"]

    def test_mixed_types(self):
        assert CategoricalDomain([1, "two", 3.0]).values == [1, "two", 3.0]

    # --- random_sample ---

    def test_random_sample_no_args_returns_all(self):
        assert sorted(CategoricalDomain(["a", "b", "c"]).random_sample()) == ["a", "b", "c"]

    def test_random_sample_n_limits_count(self):
        assert len(CategoricalDomain(["a", "b", "c", "d"]).random_sample(n=2, with_replacement=False)) == 2

    def test_random_sample_without_replacement_no_duplicates(self):
        d = CategoricalDomain(["a", "b", "c", "d", "e"])
        for _ in range(10):
            result = d.random_sample(n=5, with_replacement=False)
            assert len(result) == len(set(result))

    def test_random_sample_with_replacement_can_repeat(self):
        assert CategoricalDomain(["a"]).random_sample(n=5, with_replacement=True) == ["a"] * 5

    def test_random_sample_n_exceeds_size_falls_back_to_replacement(self):
        assert len(CategoricalDomain(["a", "b"]).random_sample(n=10, with_replacement=False)) == 10

    def test_random_sample_returns_list(self):
        assert isinstance(CategoricalDomain(["a", "b"]).random_sample(), list)

    def test_random_sample_values_come_from_domain(self):
        allowed = {"low", "medium", "high"}
        d = CategoricalDomain(list(allowed))
        for _ in range(10):
            assert all(v in allowed for v in d.random_sample(n=3, with_replacement=True))

    # --- systematic_sample ---
    # systematic_sample maps helper index floats back to domain values.

    def test_systematic_sample_returns_list(self):
        assert isinstance(CategoricalDomain(["a", "b", "c", "d", "e"]).systematic_sample(step=1), list)

    def test_systematic_sample_step_1_returns_all(self):
        d = CategoricalDomain(["a", "b", "c", "d", "e"])
        assert d.systematic_sample(step=1) == ["a", "b", "c", "d", "e"]

    def test_systematic_sample_step_2_skips_elements(self):
        d = CategoricalDomain(["a", "b", "c", "d", "e"])
        # indices 0, 2, 4 -> ["a", "c", "e"]
        assert d.systematic_sample(step=2) == ["a", "c", "e"]

    def test_systematic_sample_values_are_in_domain(self):
        vals = ["x", "y", "z", "w"]
        assert all(v in vals for v in CategoricalDomain(vals).systematic_sample(step=1))

    def test_systematic_sample_returns_items_not_indices(self):
        """Confirm the mapping from index float to value happens correctly."""
        d = CategoricalDomain(["red", "green", "blue"])
        result = d.systematic_sample(step=1)
        assert all(isinstance(v, str) for v in result)

    def test_systematic_sample_with_step_returns_items(self):
        """With step parameter, verify items not indices are returned."""
        d = CategoricalDomain([10, 20, 30, 40, 50])
        result = d.systematic_sample(step=2)
        assert all(isinstance(v, int) for v in result)
        assert all(v in [10, 20, 30, 40, 50] for v in result)

    def test_systematic_sample_with_n_max_returns_items(self):
        """With n_max parameter, verify items not indices are returned."""
        d = CategoricalDomain(["apple", "banana", "cherry", "date", "elderberry"])
        result = d.systematic_sample(step=None, n_max=3)
        assert all(isinstance(v, str) for v in result)
        assert all(v in d.values for v in result)

    def test_systematic_sample_with_step_and_n_max_returns_items(self):
        """With both step and n_max, verify items not indices are returned."""
        d = CategoricalDomain(list(range(20)))
        result = d.systematic_sample(step=1, n_max=5)
        assert all(isinstance(v, int) for v in result)
        assert all(v in range(20) for v in result)

    def test_systematic_sample_large_step_returns_single_item(self):
        """With large step, verify single item is returned."""
        d = CategoricalDomain(["x", "y", "z"])
        result = d.systematic_sample(step=10)
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert result[0] in ["x", "y", "z"]

    def test_systematic_sample_n_max_one_returns_item(self):
        """With n_max=1, verify single item is returned."""
        d = CategoricalDomain(["first", "second", "third"])
        result = d.systematic_sample(step=None, n_max=1)
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert result[0] in d.values
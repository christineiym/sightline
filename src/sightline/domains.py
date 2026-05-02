"""
Valid Domain types for Characteristics.

AI use disclosure: Initial implementation of random_sample of CategoricalDomain was drafted by GPT-5.1 and revised manually line-by-line.
"""

from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic
import copy
import random
import numpy as np
from fractions import Fraction


T = TypeVar("T")

class Domain(ABC, Generic[T]):
    @abstractmethod
    def random_sample(self, *args, **kwargs) -> List[T]:
        """Draw a random sample, assuming a uniform distribution of provided values.
        
        Note that the values currently return in non-random (sorted) order.
        """
        pass

    @abstractmethod
    def systematic_sample(self, *args, **kwargs) -> List[T]:
        """Draw a systematic sample from this domain of at most n_max equidistant elements, separated by some multiple of step.
        Default behavior is to start at min, but if step is negative, elements will start at max and proceed downward.
        
        For more details, see Domain._systematic_sample_from_range.
        """
        pass

    def _systematic_sample_from_range(self, min, max, step: int = None, n_max: int = None) -> List[T]:
        """Draw a systematic sample of at most n_max equidistant elements, separated by some multiple of step.
        Default behavior is to start at min, but if step is negative, elements will start at max and proceed downward. Stop is included.
        
        - Case 1: step and n_max are both provided.
        - Case 2: If only n_max is provided, step is calculated by dividing the domain into n_max - 1 units such that exactly n_max elements (including min and max) are returned.
        - Case 3: If only step is provided, all values in the domain that are a multiple of step are returned.
        - Case 4: If neither argument is supplied, step is calculated to be 1 divided by the denominator of the integer ratio equivalent to the fractional portion of max - min,
            or 1 if the fractional portion is 0. Note that this differs from binning, in the sense that the sample size is not constrained by heuristics for n_max for graphs.
        
        Case 4 Examples:
            Example A: 7 - 2.65 = 5.35; 0.35 is the fractional portion which simplifies to 7/20; denominator is 20, and 1/20 = 0.05. 
            Example B: 100 - 50 = 50; the fractional portion is 0, so the resulting sample will count upwards from 50.
            Example C: 5.3 - 1.3 = 4.0; the fractional portion is also 0, but 1.3 will offset the resulting sample will from whole numbers.
        """
        if n_max is not None:
            if n_max <= 0:
                raise ValueError("n_max must be a positive integer (positive float values will be rounded)")
        if step == 0:
            return[min]
        if max < min:
            temp = min
            min = max
            max = temp
        
        sample_min = min
        sample_max = max
        if step is not None and step < 0:
            sample_min = -min
            sample_max = -max
        sample_step = step
        sample_n_max = round(n_max) if n_max is not None else None

        if sample_n_max == 1:
            return [sample_min]
        
        if step is not None and n_max is not None:  # Case 1
            raw_n = ((sample_max - sample_min) // step) + 1
            scale_factor = raw_n // (sample_n_max - 1)
            sample_step = step * scale_factor
        elif step is None and n_max is not None:  # Case 2
            sample_step = (max - min) / (sample_n_max - 1)
        elif step is not None and n_max is None:  # Case 3
            pass
        else:  # Case 4
            fractional_part = (sample_max - sample_min) % 1
            simplified_fraction = Fraction.from_float(fractional_part).limit_denominator()
            sample_step = float(Fraction(1, simplified_fraction.denominator).limit_denominator())
        
        if sample_step == 0:
            return [sample_min]
        
        num = (round(((sample_max - sample_min) / sample_step)) + 1)
        sampled_values = np.linspace(start=sample_min, stop=sample_max, num=num).tolist()

        return sampled_values


class ContinuousDomain(Domain[T]):
    def __init__(self, min: T, max: T):
        """Initialize a ContinuousDomain with a given min and max."""
        self._min = min
        self._max = max
    
    @property
    def min(self) -> T:
        return self._min

    @property
    def max(self) -> T:
        return self._max

    def random_sample(self, n) -> List[T]:
        """Draw random samples from a uniform distribution. Replacement is possible, but unlikely for continuous domains."""
        return np.random.uniform(self.min, self.max, size=n).tolist()

    def systematic_sample(self, step, n_max):
        return list(super()._systematic_sample_from_range(self.min, self.max, step, n_max))


class DiscreteRangeDomain(Domain[T]):  # should probably make this an int
    def __init__(self, min: T, max: T, unit: T = 1):
        """Initialize a DiscreteRangeDomain where values start at min and increase by step to max (or as close to it as possible)."""
        self._min = min
        self._max = max
        self._unit = unit
    
    @property
    def min(self) -> T:
        return self._min

    @property
    def max(self) -> T:
        return self._max
    
    @property
    def unit(self) -> T:
        return self._unit

    def random_sample(self, n: int = None, with_replacement: bool = True) -> List[T]:
        """Return a random sample of values from the range. If n is not provided, return all values."""
        list_length = (self.max - self.min) // self.unit
        indices = super()._systematic_sample_from_range(0, self.unit, list_length)

        chosen_indices = list(indices).copy()
        if n is not None:
            chosen_indices = random.choices(indices, k=n) if with_replacement or n > len(chosen_indices) \
                else random.sample(indices, k=n)
        return [index * self.unit for index in chosen_indices]

    def systematic_sample(self, step: int = None, n_max: int = None) -> List[T]:
        """Return values evenly spaced by step across the numerical domain.
        
        When provided, step is rounded to the nearest multiple of this domain's actual step.
        Otherwise, it is assumed to equal the domain's step.
        """
        return list(super()._systematic_sample_from_range(self.min, self.max, step, n_max))
    

class CategoricalDomain(Domain[T]):
    def __init__(self, values: list[T]):
        """Initialize a CategoricalDomain with the provided values."""
        self._values = copy.deepcopy(values)
    
    @property
    def values(self) -> list[T]:
        return copy.deepcopy(self._values)

    def random_sample(self, n: int = None, with_replacement: bool = True) -> List[T]:
        if n is None:
            return list(self._values)

        if with_replacement or n > len(self._values):
            return random.choices(self._values, k=n)
        return random.sample(self._values, k=n)

    def systematic_sample(self, step: int = None, n_max: int = None) -> List[T]:
        """Return evenly indexed samples."""
        sample_step = round(step) if step is not None else 1
        indices = super()._systematic_sample_from_range(0, len(self._values) - 1, sample_step, n_max)
        return [self._values[int(i)] for i in indices]
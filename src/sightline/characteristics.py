"""
Tunable image parameters.

AI use disclosure: Initial implementation of subclasses was drafted by GPT-5.1 and revised manually line-by-line.
"""
import copy
from abc import ABC, abstractmethod
from typing import TypeVar, Iterable, Generic
from sightline.domains import Domain, ContinuousDomain, DiscreteRangeDomain, CategoricalDomain

T = TypeVar('T')

class Characteristic(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, name: str, domain: Domain[T], default: T, scope: str): 
        self._name: str = name
        self._domain: Domain[T] = domain
        self._default: T = default
        self._scope: str = scope

    @property
    def name(self) -> str:
        return self._name

    @property
    def domain(self) -> Domain[T]:
        return self._domain
    
    @property
    def default(self) -> T:
        return self._default
    
    @property
    def scope(self) -> str:
        return self._scope
    
    def get_random_sample(self, *args, **kwargs):
        """
        Return n randomly sampled discrete inputs. 
        When n is not specified, return the entire domain.
        When n exceeds domain size, ignore with_replacement.
        
        :param self: Class reference.
        :param n: Number of values requested.
        :type n: int
        :param with_replacement: Whether or not the same value can be given more than once.
        :type with_replacement: bool
        """
        return self._domain.random_sample(*args, **kwargs)

    def get_systematic_sample(self, *args, **kwargs):
        """
        Return uniformly sampled discrete inputs, up to and including max (when True).
        """
        return self._domain.systematic_sample(*args, **kwargs)


class ContinuousCharacteristic(Characteristic):
    def __init__(self, name: str, min, max, default, scope: str):
        domain = ContinuousDomain(min=min, max=max)
        super().__init__(name, domain, default, scope)

    @property
    def min(self) -> int:
        return self._domain.min
    
    @property
    def max(self) -> int:
        return self._domain.max


class DiscreteRangeCharacteristic(Characteristic):
    def __init__(self, name: str, min, max, unit, default, scope: str):
        domain = DiscreteRangeDomain(min=min, max=max, unit=unit)
        super().__init__(name, domain, default, scope)
    
    @property
    def min(self) -> int:
        return self._domain.min
    
    @property
    def max(self) -> int:
        return self._domain.max
    
    @property
    def unit(self) -> int:
        return self._domain.unit


class CategoricalCharacteristic(Characteristic):
    def __init__(self, name: str, values: Iterable[T], default, scope: str):
        domain = CategoricalDomain(values=values)
        super().__init__(name, domain, default, scope)
    
    @property
    def values(self) -> list[T]:
        return copy.deepcopy(self._domain.values)
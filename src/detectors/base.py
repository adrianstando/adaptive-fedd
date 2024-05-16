import abc
from river.base import DriftDetector
from typing import Union, List


class VirtualDriftDetector(DriftDetector):
    @abc.abstractmethod
    def update(self, x: Union[int, float, List]) -> DriftDetector:
        pass

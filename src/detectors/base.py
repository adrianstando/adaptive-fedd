import abc
import pandas as pd
from river.base import DriftDetector
from typing import Union, List


class VirtualDriftDetector(DriftDetector):
    @abc.abstractmethod
    def update(self, x: Union[int, float, List]) -> None:
        pass

class FeatureExtractor:
    @abc.abstractmethod
    def extract_features(self, time_series: pd.DataFrame) -> List:
        pass

class BasicDriftDetector:
    @abc.abstractmethod
    def add_training_elements(self, values: list) -> None:
        pass

    @abc.abstractmethod
    def detect(self, value: float) -> bool:
        pass
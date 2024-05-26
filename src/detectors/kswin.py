from river.drift import KSWIN as KSWINRiver
from typing import Union, List

from src.detectors.base import BasicDriftDetector, VirtualDriftDetector


class KSWIN(BasicDriftDetector, VirtualDriftDetector):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kswin = KSWINRiver(*args, **kwargs)

    def add_training_elements(self, values: list) -> None:
        for v in values:
            self.kswin.update(v)

    def detect(self, value: float) -> bool:
        self.kswin.update(value)
        return self.kswin.drift_detected
    
    def update(self, x: Union[int, float, List]) -> None:
        if isinstance(x, list):
            for elem in x:
                self.kswin.update(elem)
        else:
            self.kswin.update(x)

    @property
    def drift_detected(self) -> bool:
        return self.kswin.drift_detected
    
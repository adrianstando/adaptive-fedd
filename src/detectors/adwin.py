from river.drift import ADWIN as ADWINRiver
from typing import Union, List

from src.detectors.base import BasicDriftDetector, VirtualDriftDetector


class ADWIN(BasicDriftDetector, VirtualDriftDetector):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.adwin = ADWINRiver(*args, **kwargs)

    def add_training_elements(self, values: list) -> None:
        for v in values:
            self.adwin.update(v)

    def detect(self, value: float) -> bool:
        self.adwin.update(value)
        return self.adwin.drift_detected
    
    def update(self, x: Union[int, float, List]) -> None:
        if isinstance(x, list):
            for elem in x:
                self.adwin.update(elem)
        else:
            self.adwin.update(x)

    @property
    def drift_detected(self) -> bool:
        return self.adwin.drift_detected
    
    def clone(self, new_params: dict | None = None, include_attributes=False):
        out = super().clone(new_params, include_attributes)
        out.adwin = self.adwin.clone()
        return out
    
from typing import Union, List

from src.detectors.base import BasicDriftDetector, VirtualDriftDetector


class NoDrift(BasicDriftDetector, VirtualDriftDetector):
    def __init__(self):
        super().__init__()

    def add_training_elements(self, values: list) -> None:
        return

    def detect(self, value: float) -> bool:
        return
    
    def update(self, x: Union[int, float, List]) -> None:
        return

    @property
    def drift_detected(self) -> bool:
        return False
    
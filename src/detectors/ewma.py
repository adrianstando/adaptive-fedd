import numpy as np

from typing import Optional, Union, List

from src.detectors.base import BasicDriftDetector, VirtualDriftDetector


class EWMA(BasicDriftDetector, VirtualDriftDetector):
    def __init__(self, Lambda: float = 0.2, drift_threshold: float = 3, train_size = Optional[int]):
        super().__init__()
        self.Lambda = Lambda
        self.drift_threshold = drift_threshold

        self.initial_average = 0
        self.initial_std = 0
        self.ewma_std = 0
        self.ewma = 0
        self.time = 0

        self.train_size = train_size
        self._queue = []
        self._last_detection = False
        self._fitted = False

    def add_training_elements(self, values: list) -> None:
        self.initial_average = np.mean(values)
        self.initial_std = np.std(values)
        self.ewma = self.initial_average
        self._fitted = True

    def detect(self, value: float) -> bool:
        self.update_ewma(value)
        result = bool(self.ewma > self.initial_average + self.drift_threshold * self.ewma_std)
        self._last_detection = result
        return result

    def update_ewma(self, distance: float) -> None:
        self.ewma = (1 - self.Lambda) * self.ewma + self.Lambda * distance
        self.time += 1

        parte1 = self.Lambda / (2 - self.Lambda)
        parte2 = 1 - self.Lambda
        parte3 = 2 * self.time
        parte4 = 1 - (parte2 ** parte3)

        parte5 = (parte1 * parte4)
        self.ewma_std = np.sqrt(float(parte5)) * self.initial_std
    
    def update(self, x: Union[int, float, List]) -> None:
        if not self._fitted:
            if isinstance(x, list):
                for elem in x:
                    self._queue.append(elem)
            else:
                self._queue.append(x)

            if self.train_size is not None:
                if len(self._queue) >= self.train_size: # type: ignore
                    self.add_training_elements(self._queue)
                    self._queue = []
                
        else:
            if isinstance(x, list):
                for elem in x:
                    self.detect(elem)
            else:
                self.detect(x)

    @property
    def drift_detected(self) -> bool:
        return self._last_detection
    
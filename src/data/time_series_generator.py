import pandas as pd
import numpy as np

from abc import abstractmethod


class TimeSeriesGenerator:
    @abstractmethod
    def generate(self) -> pd.DataFrame:
        ...

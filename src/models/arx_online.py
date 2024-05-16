from river import time_series
from river import base


class AROnline(time_series.SNARIMAX):
    def __init__(
        self,
        p: int,
        regressor: base.Regressor | None = None
    ):
        super().__init__(p, 0, 0, 1, 0, 0, 0, regressor)

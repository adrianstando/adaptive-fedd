import pandas as pd
import numpy as np

from src.data import TimeSeriesGenerator

from pydantic import BaseModel, Field
from typing import Optional, Union, Literal, List, Dict, Callable


class AutoregressiveSeasonalParam(BaseModel):
    sigma: float = 1
    # ar_param = [alpha_{t-1}, alpha_{t-2}, ...]
    ar_param: np.ndarray = Field(default_factory=lambda: np.array([1, -1]))
    length: int = 1000
    constant: float = 0
    # drift_switch_type equal to None = abrupt drift
    drift_switch_type: Optional[Union[Literal['incremental'], Literal['gradual']]] = None
    drift_switch_length: int = 0
    seasonal_function: Callable[[Union[pd.Timestamp, int]], float] = lambda x: 0

    class Config:
        arbitrary_types_allowed = True


class AutoregressiveSeasonalModel(TimeSeriesGenerator):
    def __init__(self, params: List[Dict]) -> None:
        """Autoregressive with Exogenous input (seasonal) Model

        Parameters
        ----------
        params : List
            List of dictionaries following AutoregressiveSeasonalParam schema.
        """
        super().__init__()
        self.params = []

        try:
            self.params = [
                AutoregressiveSeasonalParam(**elem)
                for elem in params
            ]
        except:
            raise Exception('Incorrect params')
    
    def generate(self) -> pd.DataFrame:
        def calculate_ar(ar, y, cnst):
            history = np.array(y) - cnst
            return np.sum(ar * history)

        def calculate_val(ar, y, cnst, sigma, seasonal_func, i):
            val = 0
            val += cnst
            val += calculate_ar(ar, y, cnst)
            val += seasonal_func(i)
            val += np.random.normal(scale=sigma)
            return val

        mimimal_filled = self.params[0].length
        start = mimimal_filled
        end = mimimal_filled
        n = int(np.sum([d.length for d in self.params]))
        y = np.zeros(n + mimimal_filled)

        for k, param in enumerate(self.params):
            end += param.length
            ar_length = len(param.ar_param)

            for i in range(start, end):
                if param.drift_switch_type is not None and param.drift_switch_length > i - start:
                    if param.drift_switch_type == 'incremental':
                        weight = (i - start) / param.drift_switch_length

                        current_val = calculate_val(
                            ar=np.array(param.ar_param)[::-1],
                            y=y[i - ar_length:i],
                            cnst=param.constant,
                            sigma=param.sigma,
                            seasonal_func=param.seasonal_function,
                            i=i
                        )

                        ar_length_old = len(self.params[k - 1].ar_param)
                        old_val = calculate_val(
                            ar=np.array(self.params[k - 1].ar_param)[::-1],
                            y=y[i - ar_length_old:i],
                            cnst=self.params[k - 1].constant,
                            sigma=self.params[k - 1].sigma,
                            seasonal_func=self.params[k - 1].seasonal_function,
                            i=i
                        )

                        y[i] = weight * current_val + (1 - weight) * old_val
                    elif param.drift_switch_type == 'gradual':
                        p = (i - start) / param.drift_switch_length
                        indicator = np.random.binomial(1, p, size=1)[0]
                        if indicator == 0:
                            ar_length_old = len(self.params[k - 1].ar_param)
                            old_val = calculate_val(
                                ar=np.array(self.params[k - 1].ar_param)[::-1],
                                y=y[i - ar_length_old:i],
                                cnst=self.params[k - 1].constant,
                                sigma=self.params[k - 1].sigma,
                                seasonal_func=self.params[k - 1].seasonal_function,
                                i=i
                            )
                            y[i] = old_val
                        else:
                            y[i] = calculate_val(
                                ar=np.array(param.ar_param)[::-1],
                                y=y[i - ar_length:i],
                                cnst=param.constant,
                                sigma=param.sigma,
                                seasonal_func=param.seasonal_function,
                                i=i
                            )
                    else:
                        raise KeyError
                else:
                    y[i] = calculate_val(
                        ar=np.array(param.ar_param)[::-1],
                        y=y[i - ar_length:i],
                        cnst=param.constant,
                        sigma=param.sigma,
                        seasonal_func=param.seasonal_function,
                        i=i
                    )
            
            start += param.length

        return pd.DataFrame({
            'timestamp': np.arange(n).tolist(),
            'value': y[mimimal_filled:]
        })
    
import pandas as pd
import numpy as np

from src.data import TimeSeriesGenerator

from pydantic import BaseModel, Field
from typing import Optional, Union, Literal, List, Dict


class NonlinearParam(BaseModel):
    sigma: float = 1
    # ar_param = [alpha_{t-1}, alpha_{t-2}, ...]
    ar_param: np.ndarray = Field(default_factory=lambda: np.array([1, -1]))
    ar_exp_param: np.ndarray = Field(default_factory=lambda: np.array([]))
    length: int = 1000
    constant: float = 0
    # drift_switch_type equal to None = abrupt drift
    drift_switch_type: Optional[Union[Literal['incremental'], Literal['gradual']]] = None
    drift_switch_length: int = 0

    class Config:
        arbitrary_types_allowed = True


class NonlinearModel(TimeSeriesGenerator):
    def __init__(self, params: List[Dict]) -> None:
        """Nonlinear Model

        Parameters
        ----------
        params : List
            List of dictionaries following NonlinearParam schema.
        """
        super().__init__()
        self.params = []

        try:
            self.params = [
                NonlinearParam(**elem)
                for elem in params
            ]
        except:
            raise Exception('Incorrect params')
    
    def generate(self) -> pd.DataFrame:
        def calculate_ar(ar, y, cnst):
            if len(ar) == 0:
                return 0
            history = np.array(y) - cnst
            return np.sum(ar * history)
        
        def calculate_exp_ar(ar, y, cnst):
            if len(ar) == 0:
                return 0
            history = np.array(y) - cnst
            last = y[-1] if y[-1] != 0 else 0.1
            return np.sum(ar * history) * 1 / (1 - np.exp(-10 * (last - cnst)))

        def calculate_val(ar, ar_exp, y, cnst, sigma):
            val = 0
            val += cnst
            if len(ar) > 0:
                val += calculate_ar(ar, y[-len(ar):], cnst)
            if len(ar_exp) > 0:
                val += calculate_exp_ar(ar_exp, y[-len(ar_exp):], cnst)
            val += np.random.normal(scale=sigma)
            return val
        
        def calculate_maximal_lag(param):
            return max(len(param.ar_param), len(param.ar_exp_param))

        mimimal_filled = self.params[0].length
        start = mimimal_filled
        end = mimimal_filled
        n = int(np.sum([d.length for d in self.params]))
        y = np.zeros(n + mimimal_filled)

        for k, param in enumerate(self.params):
            end += param.length
            ar_length = calculate_maximal_lag(param)

            for i in range(start, end):
                if param.drift_switch_type is not None and param.drift_switch_length > i - start:
                    if param.drift_switch_type == 'incremental':
                        weight = (i - start) / param.drift_switch_length

                        current_val = calculate_val(
                            ar=np.array(param.ar_param)[::-1],
                            ar_exp=np.array(param.ar_exp_param)[::-1],
                            y=y[i - ar_length:i],
                            cnst=param.constant,
                            sigma=param.sigma
                        )

                        ar_length_old = calculate_maximal_lag(self.params[k - 1])
                        old_val = calculate_val(
                            ar=np.array(self.params[k - 1].ar_param)[::-1],
                            ar_exp=np.array(param.ar_exp_param)[::-1],
                            y=y[i - ar_length_old:i],
                            cnst=self.params[k - 1].constant,
                            sigma=self.params[k - 1].sigma
                        )

                        y[i] = weight * current_val + (1 - weight) * old_val
                    elif param.drift_switch_type == 'gradual':
                        p = (i - start) / param.drift_switch_length
                        indicator = np.random.binomial(1, p, size=1)[0]
                        if indicator == 0:
                            ar_length_old = calculate_maximal_lag(self.params[k - 1])
                            old_val = calculate_val(
                                ar=np.array(self.params[k - 1].ar_param)[::-1],
                                ar_exp=np.array(param.ar_exp_param)[::-1],
                                y=y[i - ar_length_old:i],
                                cnst=self.params[k - 1].constant,
                                sigma=self.params[k - 1].sigma
                            )
                            y[i] = old_val
                        else:
                            y[i] = calculate_val(
                                ar=np.array(param.ar_param)[::-1],
                                ar_exp=np.array(param.ar_exp_param)[::-1],
                                y=y[i - ar_length:i],
                                cnst=param.constant,
                                sigma=param.sigma
                            )
                    else:
                        raise KeyError
                else:
                    y[i] = calculate_val(
                        ar=np.array(param.ar_param)[::-1],
                        ar_exp=np.array(param.ar_exp_param)[::-1],
                        y=y[i - ar_length:i],
                        cnst=param.constant,
                        sigma=param.sigma
                    )
            
            start += param.length

        return pd.DataFrame({
            'timestamp': np.arange(n).tolist(),
            'value': y[mimimal_filled:]
        })
    
import pandas as pd

from abc import abstractmethod
from typing import Union, List


def check_dataframe(df):
    if not {'timestamp', 'value'}.issubset(set(df.columns)):
            raise Exception('Wrong column names in the dataframe!')


class Experiment:
    def __init__(self, data: pd.DataFrame) -> None:
        check_dataframe(data)        
        self.data = data
        self.max_len = data.shape[0]
        self.current_idx = 0
        self._timestamp_data_type = 'timestamp' if isinstance(data.iloc[0]['timestamp'], pd.Timestamp) else 'int'
        self._stride = data['timestamp'].diff().min() if isinstance(data.iloc[0]['timestamp'], pd.Timestamp) else 1

        self.predictions = pd.DataFrame(columns=['origin', 'timestamp', 'value'])

    def add_predictions(self, origin: Union[pd.Timestamp, int], values: pd.DataFrame):
        check_dataframe(values)
        out = values.copy()
        out['origin'] = origin
        out = out[['origin', 'timestamp', 'value']]

        if self.predictions.shape[0] == 0:
            self.predictions = out
        else:
            self.predictions = pd.concat([self.predictions, out])

    def future_timestamps(self, horizon: int) -> List[Union[pd.Timestamp, int]]: # type: ignore
        # normal situation
        if self.current_idx + horizon + 1 < self.max_len:
            return self.data.iloc[self.current_idx + 1 : self.current_idx + 1 + horizon]['timestamp'].tolist()
        
        # part of horizon is outside
        elif self.current_idx + 1 + horizon >= self.max_len > self.current_idx + 1:
            first_part = self.data.iloc[self.current_idx + 1 : self.max_len]['timestamp'].tolist()

            if self._timestamp_data_type == 'timestamp':
                second_part = pd.date_range(
                    start=first_part[-1] + self._stride,
                    periods=horizon-len(first_part),
                    freq=self._stride # type: ignore
                ).tolist()
            else:
                second_part = [first_part[-1] + i + 1 for i in range(horizon-len(first_part))]
            
            return first_part + second_part

        # the whole horizon is outside
        else:
            if self._timestamp_data_type == 'timestamp':
                return pd.date_range(
                    start=self.data.iloc[-1]['timestamp'] + self._stride, # type: ignore
                    periods=horizon,
                    freq=self._stride # type: ignore
                ).tolist()
            else:
                return [self.current_idx + i + 1 for i in range(horizon)]
        

    def step(self) -> None:
        self._step(self.current_idx, self.data.iloc[self.current_idx]['timestamp'], self.data.iloc[self.current_idx]['value'])
        self.current_idx += 1

    @abstractmethod
    def _step(self, idx: int, timestamp: Union[pd.Timestamp, int], value: Union[float, int]) -> None:
        ...

import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Optional, Union, Any, Sequence
from typing import Dict as DictLike
from torch.utils.data import Dataset


def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)

class SimpleReplay(ABC):
    def __init__(self, max_size: int, field_specs: Optional[DictLike]=None, *args, **kwargs):
        super().__init__()
        self.field_specs = {}
        self.fields = {}
        
        self._max_size = int(max_size)
        self._size = 0
        self.add_fields(field_specs)
                
    def __len__(self):
        return self._size
        
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def add_fields(self):
        raise NotImplementedError
    
    @abstractmethod
    def add_sample(self):
        raise NotImplementedError
    
    @abstractmethod
    def random_batch(self):
        raise NotImplementedError


class TransitionSimpleReplay(SimpleReplay, Dataset):
    def __init__(self, max_size: int, field_specs: Optional[DictLike]=None, *args, **kwargs):
        SimpleReplay.__init__(self, max_size, field_specs, *args, **kwargs)
        Dataset.__init__(self)
        self._size = 0
        self._count = 0
        
    def reset(self):
        self._pointer = self._size = 0
        self.fields = self.fields or {}
        for _key, _specs in self.field_specs.items():
            initializer = _specs.get("initialzier", np.zeros)
            self.fields[_key] = initializer(shape=[self._max_size, ]+list(_specs["shape"]), dtype=_specs["dtype"])
            
    def add_fields(self, new_field_specs: Optional[DictLike]=None):
        new_field_specs = new_field_specs or {}
        self.fields = self.fields or {}
        for _key, _specs in new_field_specs.items():
            _old_specs = self.field_specs.get(_key, None)
            if _old_specs is None or _old_specs != _specs:
                self.field_specs[_key] = _specs
                initializer = _specs.get("initializer", np.zeros)
                self.fields[_key] = initializer(shape=[self._max_size, ]+list(_specs["shape"]), dtype=_specs["dtype"])
                
    def add_sample(self, key_or_dict: Union[str, DictLike], data: Optional[Any]=None):
        if isinstance(key_or_dict, str):
            key_or_dict = {key_or_dict: data}
        unsqueeze = None
        data_len = None
        for _key, _data in key_or_dict.items():
            if _key in self.field_specs.keys():
                if unsqueeze is None:
                    unsqueeze = len(_data.shape) == len(self.field_specs[_key]["shape"])
                    data_len = 1 if unsqueeze else _data.shape[0]
                    index_to_go = np.arange(self._pointer, self._pointer + data_len) % self._max_size
                    # if _data.shape[-1] == data_len:
                    #     _data = _data.reshape((data_len, -1))
                # unsqueeze = None
                self.fields[_key][index_to_go] = _data
        self._pointer = (self._pointer + data_len) % self._max_size
        self._size = min(self._size + data_len, self._max_size)
        
    def random_batch(self, batch_size: Optional[int]=None, fields: Optional[Sequence[str]]=None, return_idx: bool=False):
        if len(self) == 0:
            batch_data, batch_idx = None, None
        else:
            if batch_size is None:
                batch_idx = np.arange(0, len(self))
                np.random.shuffle(batch_idx)
            else:
                batch_idx = np.random.randint(0, len(self), batch_size)
            if fields is None:
                fields = self.field_specs.keys()
            batch_data = {
                _key:self.fields[_key][batch_idx] for _key in fields
            }
        return (batch_data, batch_idx) if return_idx else batch_data

    def __getitem__(self, index):
        sample = {
            _key:self.fields[_key][index] for _key in self.field_specs.keys()
        }
        return sample

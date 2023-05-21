import abc

import numpy as np


class Located(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_location(self) -> np.ndarray | float:
        pass


class ValuedLocated(Located, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self):
        pass

import numpy as np

from base import Located


class Element1D(Located):
    def __init__(self, loc: float):
        self.loc = loc

    def get_location(self) -> np.ndarray | float:
        return self.loc

    def __repr__(self) -> str:
        return f"Sample1D({self.loc:.4f})"


class Element2D(Located):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def get_location(self) -> np.ndarray | float:
        return np.array([self.x, self.y], dtype=np.float64)

    def __repr__(self) -> str:
        return f"Sample2D({self.x:.4f}, {self.y:.4f})"

import numpy as np

from base import Located
from base import ValuedLocated


class MeasuringParam:
    def __init__(self, distance_weight: float, value_weight: float = 0):
        assert (distance_weight + value_weight) < 1

        self.distance_weight = distance_weight
        self.value_weight = value_weight
        self.operation_weight = 1 - distance_weight - value_weight


def measure(pop1: list[Located], pop2: list[Located], param: MeasuringParam | None = None) -> float:
    if len(pop1) == 0 and len(pop2) == 0:
        return 0
    elif len(pop1) <= len(pop2):
        smaller = pop1
        larger = pop2
    else:
        smaller = pop2
        larger = pop1

    is_valued = isinstance(larger[0], ValuedLocated)
    if param is None:
        if is_valued:
            param = MeasuringParam(1/3, 1/3)
        else:
            param = MeasuringParam(0.5, 0.0)

    if len(smaller) == 0:
        return _measure_from_o(larger, param)

    loc_dist = _measure_location_dist(smaller, larger)
    ope_dist = len(larger) - len(smaller)

    if is_valued:
        val_dist = _measure_value_dist(smaller, larger)
    else:
        val_dist = 0

    total_dist_elements = _find_proximal(loc_dist * param.distance_weight + val_dist * param.value_weight)

    return np.sum(total_dist_elements) + ope_dist * param.operation_weight


def _measure_from_o(pop: list[Located], param: MeasuringParam) -> float:
    loc_dist = sum([sample.get_location() for sample in pop])
    ope_dist = len(pop)
    if isinstance(pop[0], ValuedLocated):
        val_dist = sum([sample.get_value() for sample in pop])
    else:
        val_dist = 0
    return loc_dist * param.distance_weight + val_dist * param.value_weight + ope_dist * param.operation_weight


def _measure_location_dist(smaller: list[Located], larger: list[Located]) -> np.ndarray:
    loc_smaller = np.array([sample.get_location() for sample in smaller], dtype=np.float64)
    loc_larger = np.array([sample.get_location() for sample in larger], dtype=np.float64)

    if loc_smaller.ndim == 1:
        mat_size = (len(loc_larger), len(loc_smaller))
        _loc_smaller_mat = np.ones(mat_size, dtype=np.float64) * loc_smaller
        _loc_larger_mat = np.ones(mat_size, dtype=np.float64) * loc_larger[:, np.newaxis]
        dist_matrix = np.abs(_loc_larger_mat - _loc_smaller_mat)
    else:
        mat_size = (len(loc_larger), len(loc_smaller), len(loc_smaller[0]))
        _loc_smaller_mat = np.ones(mat_size, dtype=np.float64) * loc_smaller
        _loc_larger_mat = np.ones(mat_size, dtype=np.float64) * loc_larger[:, np.newaxis]
        dist_matrix = np.sqrt(np.sum(np.square(_loc_larger_mat - _loc_smaller_mat), axis=2))

    return dist_matrix


def _measure_value_dist(smaller: list[ValuedLocated], larger: list[ValuedLocated]) -> np.ndarray:
    val_smaller = np.array([sample.get_value() for sample in smaller], dtype=np.float64)
    val_larger = np.array([sample.get_value() for sample in larger], dtype=np.float64)

    mat_size = (len(val_larger), len(val_smaller))
    _val_smaller_mat = np.ones(mat_size, dtype=np.float64) * val_smaller
    _val_larger_mat = np.ones(mat_size, dtype=np.float64) * val_larger[:, np.newaxis]
    return np.abs(_val_larger_mat - _val_smaller_mat)


def _find_proximal(dist_matrix: np.ndarray) -> np.ndarray:
    assert dist_matrix.ndim == 2, "Only 2-dimensional matrices supported."
    assert len(dist_matrix) >= len(dist_matrix[0]), "Only vertical or square matrices are supported."
    max_val = 1.79769313486231e+308

    h_size = len(dist_matrix[0])
    proximal_distance = np.zeros(h_size, dtype=np.float64)
    for _ in range(h_size):
        # find the most proximal
        min_index = np.argmin(dist_matrix)
        min_row = min_index // h_size
        min_col = min_index % h_size

        # log distance
        proximal_distance[min_col] = dist_matrix[min_row, min_col]

        # post process
        dist_matrix[:, min_col] = max_val

    return proximal_distance

import time

import numpy as np

import elements
import distance_measure


def log_performance(pop1_num: int, pop2_num: int, times: int = 10) -> float:
    seconds = np.zeros(times, dtype=np.float64)
    param = distance_measure.MeasuringParam(0.5)
    for i in range(times):
        pop1 = [elements.Element1D(x) for x in np.random.random(pop1_num)]
        pop2 = [elements.Element1D(x) for x in np.random.random(pop2_num)]

        start = time.time()
        _ = distance_measure.measure(pop1, pop2, param)
        seconds[i] = time.time() - start
    return float(np.mean(seconds))


if __name__ == "__main__":
    nums = [10, 30, 100, 300, 1000]
    print(f"Number of elements of the set: {nums}")
    print("unit: sec")
    print("{0:>10} | {1:>12} | {2:>24}".format("num", "mean time [s]", "time per 1 element [s]"))
    print("-"*11 + "+" + "-"*15 + "+" + "-"*25)
    for num in nums:
        mean_time = log_performance(num, num)
        time_per_a_element = mean_time / num / num
        print("{0:>10} |     {1:>.7f} |     {2:>20e}".format(num**2, mean_time, time_per_a_element))
    print("Done")

import time

import numpy as np

import elements
import distance_measure


def log_performance(pop1_num: int, pop2_num: int, times: int = 10) -> tuple[float, float]:
    seconds = np.zeros(times, dtype=np.float64)
    param = distance_measure.MeasuringParam(0.5)
    for i in range(times):
        pop1 = [elements.Element1D(x) for x in np.random.random(pop1_num)]
        pop2 = [elements.Element1D(x) for x in np.random.random(pop2_num)]

        start = time.time()
        _ = distance_measure.measure(pop1, pop2, param)
        seconds[i] = time.time() - start
    return float(np.mean(seconds)), float(np.std(seconds))


if __name__ == "__main__":
    _max_num = 14
    nums = np.arange(1, _max_num + 1) * 100
    mean_log = np.zeros(_max_num, dtype=np.float64)
    std_log = np.zeros(_max_num, dtype=np.float64)
    whole_log = np.zeros((3, _max_num), dtype=np.float64)
    
    print(f"Number of elements of the set: {nums}")
    print("unit: sec")
    print("{0:>10} | {1:>12} | {2:>24}".format("num", "mean time [s]", "time per 1 element [s]"))
    print("-"*11 + "+" + "-"*15 + "+" + "-"*25)
    for i in range(_max_num):
        mean_time, std_time = log_performance(nums[i], nums[i])
        time_per_a_element = mean_time / nums[i] / nums[i]
        print("{0:>10} |     {1:>.7f} |     {2:>20e}".format(nums[i]**2, mean_time, time_per_a_element))
        mean_log[i] = mean_time
        std_log[i] = std_time

    whole_log[0] = nums.astype(np.float64)
    whole_log[1] = mean_log
    whole_log[2] = std_log
    np.save("benchmark_log.npy", whole_log)

    print("Done")

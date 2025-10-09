import math
from functools import reduce

import numpy as np


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def multi_lcm(numbers: list[int]) -> int:
    return reduce(lcm, numbers)


def split_series(series: np.ndarray, periods: list[int], num_test_cycles: int = 1) -> tuple[np.ndarray, np.ndarray]:
    macro_cycle_length = multi_lcm(periods)
    test_length = num_test_cycles * macro_cycle_length

    if len(series) < test_length:
        raise ValueError(
            f"Длина ряда ({len(series)}) недостаточна для создания тестового набора "
            f"из {num_test_cycles} циклов по {macro_cycle_length} точек (требуется {test_length} точек)."
        )

    split_point = len(series) - test_length

    if split_point == 0:
        raise ValueError(
            f"Длина ряда ({len(series)}) в точности равна длине тестового набора. Для train набора не осталось данных."
        )

    train_series = series[:split_point]
    test_series = series[split_point:]

    print(f"Длина макро-цикла: {macro_cycle_length}")
    print(f"Длина тестового набора: {test_length} ({num_test_cycles} циклов)")
    print(f"Точка разделения: {split_point}")
    print(f"Итоговая длина train: {len(train_series)}, test: {len(test_series)}")

    return train_series, test_series

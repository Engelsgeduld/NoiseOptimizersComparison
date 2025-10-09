from typing import Any

import numpy as np
from darts import TimeSeries
from darts.utils.timeseries_generation import sine_timeseries

from series_generator.utils.series_split import split_series


class SeriesGenerator:
    def __init__(
        self,
        base_process_config: dict[str, Any] | None = None,
        trend_config: dict[str, Any] | None = None,
        seasonalities: list[dict[str, Any]] | None = None,
    ):
        self.base_process_config = base_process_config
        self.trend_config = trend_config
        self.seasonalities = seasonalities if seasonalities is not None else []

    def _create_component(self, config: dict[str, Any] | None, length: int) -> TimeSeries | None:
        if not config:
            return None

        generator_func = config.get("generator")
        params = config.get("params", {})

        if not callable(generator_func):
            raise ValueError("Конфигурация компонента должна содержать 'generator' (вызываемую функцию).")

        import inspect

        sig = inspect.signature(generator_func)

        if "length" in sig.parameters:
            component = generator_func(length=length, **params)
        else:
            t = np.arange(length)
            values = generator_func(t, **params)
            component = TimeSeries.from_values(values)

        return component

    def generate(self, length: int) -> TimeSeries:
        final_series = TimeSeries.from_values(np.zeros(length))

        base_process_component = self._create_component(self.base_process_config, length)
        if base_process_component:
            final_series = final_series + base_process_component

        trend_component = self._create_component(self.trend_config, length)
        if trend_component:
            final_series = final_series + trend_component

        for season in self.seasonalities:
            period = season.get("period")
            amplitude = season.get("amplitude", 1.0)
            if period is None:
                raise ValueError("Каждая сезонность должна иметь 'period'.")

            seasonal_component = sine_timeseries(length=length, value_amplitude=amplitude, value_frequency=1 / period)
            final_series = final_series + seasonal_component

        return final_series

    def split(
        self, total_length: int, num_test_cycles: int = 1, default_test_percent: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray]:
        full_series = self.generate(length=total_length)
        full_series_np = full_series.values().flatten()

        periods = [s["period"] for s in self.seasonalities]

        if not periods:
            print(f"Предупреждение: Сезонности не заданы. ({default_test_percent * 100}% на тест).")
            split_point = int(total_length * (1 - default_test_percent))
            train = full_series_np[:split_point]
            test = full_series_np[split_point:]
        else:
            train, test = split_series(series=full_series_np, periods=periods, num_test_cycles=num_test_cycles)

        return train, test

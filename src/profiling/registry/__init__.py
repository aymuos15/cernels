"""The profiler registry: every profileable model, keyed by name."""

from profiling.registry.base import ModelProfile
from profiling.registry.deepseek_ocr_2 import DeepseekOcr2
from profiling.registry.lfm2_5_vl_450m import Lfm2_5Vl450M
from profiling.registry.north_mini_code import NorthMiniCode

MODELS: dict[str, type[ModelProfile]] = {m.name: m for m in (DeepseekOcr2, Lfm2_5Vl450M, NorthMiniCode)}

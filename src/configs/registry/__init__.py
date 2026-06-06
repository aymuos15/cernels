"""The registry: every benchmarkable kernel config, keyed by name. Add a kernel here."""

from configs.base import Config
from configs.registry.gelu_fast import GeluFast
from configs.registry.relu import Relu
from configs.registry.rotary import Rotary
from configs.registry.silu_and_mul import SiluAndMul

CONFIGS: dict[str, type[Config]] = {c.name: c for c in (Relu, GeluFast, Rotary, SiluAndMul)}

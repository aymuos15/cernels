"""The registry: every custom kernel, keyed by name. Add a kernel here."""

from collections.abc import Callable

from kops.registry.nms import kernel as nms_kernel
from kops.registry.rope import kernel as rope_kernel

KOPS: dict[str, Callable] = {"rope": rope_kernel, "nms": nms_kernel}

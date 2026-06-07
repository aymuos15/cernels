# configs

One `Config` subclass per kernel, living in [`registry/`](registry/) and listed in [`registry/__init__.py`](registry/__init__.py). See [`docs/list.md`](../../docs/list.md) for the full table.

```bash
uv run --no-sync python -m benchmark.main <name>   # <name> = a Config.name
```

## Adding a kernel

See [how to add a config](../../docs/guide/how_to_add_a_config.md).

## custom kernels

Set `custom = staticmethod(fn)` to benchmark your own kernel (from [`src/kops/`](../kops/README.md)) as the `custom` workload alongside `op_eager` / `op_compile` / `hub` / `lib`, verified against the reference. This is how you try to *beat* the reference (and any `hub` kernel).

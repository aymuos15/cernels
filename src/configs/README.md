# configs

One YAML per kernel to benchmark. See `example.yaml` for the shape.

```bash
uv run --no-sync python src/benchmark/main.py <config>   # runs src/configs/<config>.yaml
```

## Fields

| field | meaning |
|---|---|
| `repo` | Hub kernel repo id |
| `version` | kernel version (pulled from the `v<version>` branch) |
| `dtype` | torch dtype for generated inputs (e.g. `float16`, `float32`) |
| `inputs` | list of shapes; one `torch.randn` tensor per entry |
| `baseline` | native eager reference (the `eager` workload + correctness check) |
| `op` | attribute on the loaded kernel to call |
| `out_arg` | `true` if the kernel writes into a preallocated leading `out` tensor (`k.op(out, *inputs)`), `false` if it returns the result (`k.op(*inputs)`) |

## baseline

Either a dotted import path (`torch.relu`, `torch.nn.functional.gelu`) or a bare
name of a function in `helpers.py` — used for composite references that have no
single dotted path. Add those functions to `helpers.py` as needed.

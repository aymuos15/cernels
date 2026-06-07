"""Profiling engine: three lenses over a model's forward.

1. modules  — per-module CUDA-event timing, reduced to SELF time per class (inclusive
   minus direct children), so functional work inside a module (sdpa, rope) shows up as
   that module's self time and there is no double counting. Maps to kernelize targets.
2. ops      — torch.profiler / Kineto, top CUDA ops by self device time. The
   authoritative "which kernel dominates" view.
3. inductor — torch.compile the text backbone and dump the generated Triton, to see
   what compile already fuses (and harvest a starting point).

A lens takes a `run` callable (timed) and an optional `setup` callable (untimed, e.g. a
prefill that builds the KV cache before decode is timed).
"""

import glob
import os
import re

import torch


class Runner:
    """Drives prefill and decode for a loaded model; phases are timed separately."""

    def __init__(self, model, inputs, device, decode_tokens):
        self.model = model
        self.inputs = inputs
        self.device = device
        self.n = decode_tokens
        self._pkv = None
        self._tok = None
        self._pos = 0

    @torch.no_grad()
    def prefill(self):
        return self.model(**self.inputs, use_cache=True)

    def setup_decode(self):
        """Untimed: run a prefill to establish the cache + first decode token."""
        out = self.prefill()
        self._pkv = out.past_key_values
        self._tok = out.logits[:, -1:].argmax(-1)
        self._pos = int(self.inputs["input_ids"].shape[1])

    @torch.no_grad()
    def decode(self):
        """Timed: n incremental decode steps from the cache set up by setup_decode()."""
        for _ in range(self.n):
            cache_position = torch.tensor([self._pos], device=self.device)
            out = self.model(
                input_ids=self._tok, past_key_values=self._pkv, use_cache=True, cache_position=cache_position
            )
            self._pkv = out.past_key_values
            self._tok = out.logits[:, -1:].argmax(-1)
            self._pos += 1


def profile_modules(model, run, setup=None):
    """Return [(class_name, self_ms, incl_ms, calls)] sorted by self_ms desc."""
    if setup is not None:
        setup()
    starts, ends = {}, {}
    handles = []

    def pre(m, _inp):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        starts.setdefault(m, []).append(e)

    def post(m, _inp, _out):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        ends.setdefault(m, []).append(e)

    for m in model.modules():
        handles.append(m.register_forward_pre_hook(pre))
        handles.append(m.register_forward_hook(post))
    run()
    torch.cuda.synchronize()
    for h in handles:
        h.remove()

    incl = {}
    for m, ss in starts.items():
        es = ends.get(m, [])
        incl[m] = sum(s.elapsed_time(e) for s, e in zip(ss, es))

    self_ms, incl_ms, calls = {}, {}, {}
    for m, t in incl.items():
        cname = type(m).__name__
        child = sum(incl.get(c, 0.0) for c in m.children())
        self_ms[cname] = self_ms.get(cname, 0.0) + (t - child)
        incl_ms[cname] = incl_ms.get(cname, 0.0) + t
        calls[cname] = calls.get(cname, 0) + len(starts[m])
    rows = [(c, self_ms[c], incl_ms[c], calls[c]) for c in self_ms]
    return sorted(rows, key=lambda r: r[1], reverse=True)


def profile_ops(run, setup=None, top=30):
    """Return [(op_name, self_cuda_ms, count)] sorted by self CUDA time desc."""
    from torch.profiler import ProfilerActivity, profile

    if setup is not None:
        setup()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    rows = [
        (e.key, e.self_device_time_total / 1000.0, e.count) for e in prof.key_averages() if e.self_device_time_total > 0
    ]
    return sorted(rows, key=lambda r: r[1], reverse=True)[:top]


_TRITON_NAME = re.compile(r"triton_(?:poi|red|tem|per|for|mm|ext)_[0-9a-z_]*fused[0-9a-z_]*")


def _triton_kernels(text):
    """The fused triton_* kernel names Inductor generated — they encode the fused ops
    (e.g. triton_red_fused_mean_pow_rsqrt_mul = RMSNorm)."""
    return sorted(set(_TRITON_NAME.findall(text)))


def dump_inductor(module, call, model, inductor_dir):
    """Best-effort: compile `module`, run `call` once, save generated Triton per graph
    region under inductor_dir/, with an INDEX.md manifest. Never raises.

    Returns [{region, file, kernels}] — one entry per compiled graph region.
    """
    import shutil

    os.makedirs(inductor_dir, exist_ok=True)
    shutil.rmtree("torch_compile_debug", ignore_errors=True)  # so runs[-1] is this run only
    # TORCH_COMPILE_DEBUG=1 (set in profiling/__init__ before torch import) drives the dump.
    try:
        import torch._inductor.config as ic

        ic.trace.enabled = True  # writes under ./torch_compile_debug/run_<ts>/
        ic.force_disable_caches = True  # else a warm inductor cache skips codegen -> no dump
        compiled = torch.compile(module)
        call(compiled)
        torch.cuda.synchronize()
    except Exception as exc:  # graph breaks / compile failures are informative, not fatal
        print(f"  inductor: compile/run failed ({type(exc).__name__}: {exc})")

    runs = sorted(glob.glob("torch_compile_debug/run_*"))
    entries = []
    if runs:
        for src in sorted(glob.glob(os.path.join(runs[-1], "**", "output_code.py"), recursive=True)):
            region = os.path.basename(os.path.dirname(src))  # e.g. model__7_inference_0.0
            try:
                text = open(src).read()
                shutil.copy(src, os.path.join(inductor_dir, f"{region}.py"))
            except OSError:
                continue
            graph = os.path.join(os.path.dirname(src), "fx_graph_readable.py")  # documents the region
            if os.path.exists(graph):
                shutil.copy(graph, os.path.join(inductor_dir, f"{region}.graph.py"))
            entries.append({"region": region, "file": f"{region}.py", "kernels": _triton_kernels(text)})
    _write_index(model, inductor_dir, entries)
    return entries


def _write_index(model, inductor_dir, entries):
    lines = [
        f"# Inductor-generated Triton — {model}",
        "",
        "One file per compiled graph region (`<region>.py` = generated Triton; `<region>.graph.py`",
        "= the readable FX graph). The `triton_*_fused_*` kernel names encode the fused ops —",
        "e.g. `mean_pow_rsqrt_mul` = RMSNorm, `silu_mul` = SwiGLU. If an op already appears here,",
        "Inductor fuses it and a hand kernel must beat that, not eager.",
        "",
    ]
    if not entries:
        lines.append("_No Triton generated (fell back to eager / no fused kernels)._")
    for e in entries:
        lines.append(f"## {e['file']}  ({len(e['kernels'])} fused kernels)")
        lines += [f"- {k}" for k in e["kernels"]] or ["- (none extracted)"]
        lines.append("")
    open(os.path.join(inductor_dir, "INDEX.md"), "w").write("\n".join(lines))

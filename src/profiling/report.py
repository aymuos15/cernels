"""Render a profile record (from save.py / engine.py) as human-readable tables."""

# substring -> the kernelizable op a class maps to (hint column only)
_HINTS = [
    ("rmsnorm", "rmsnorm"),
    ("layernorm", "layernorm"),
    ("rotary", "rope"),
    ("attention", "attention/sdpa"),
    ("shortconv", "causal_conv1d"),
    ("conv1d", "causal_conv1d"),
    ("mlp", "swiglu/mlp"),
    ("linear", "matmul"),
    ("gelu", "activation"),
    ("silu", "activation"),
]


def _hint(name):
    low = name.lower()
    for sub, op in _HINTS:
        if sub in low:
            return op
    return ""


def _module_table(rows):
    # Denominator = wall time of the phase = the largest inclusive time (the root module),
    # not the sum of self-times: on launch-bound decode, container self-time is mostly CPU
    # launch-gap, so self-times do not telescope to wall time. self% is "fraction of wall".
    wall = max((r[2] for r in rows), default=1.0) or 1.0
    out = [f"  {'class':28} {'self ms':>9} {'self%':>6} {'incl ms':>9} {'calls':>6}  hint"]
    for cname, self_ms, incl_ms, calls in rows:
        if self_ms < 1e-4 and incl_ms < 1e-4:
            continue
        out.append(
            f"  {cname:28} {self_ms:9.3f} {100 * self_ms / wall:5.1f}% {incl_ms:9.3f} {calls:6d}  {_hint(cname)}"
        )
    return "\n".join(out)


def _op_table(rows):
    total = sum(r[1] for r in rows) or 1.0
    out = [f"  {'op':40} {'self ms':>9} {'self%':>6} {'count':>6}"]
    for op, self_ms, count in rows:
        out.append(f"  {op[:40]:40} {self_ms:9.3f} {100 * self_ms / total:5.1f}% {count:6d}")
    return "\n".join(out)


def build_report(record):
    name = record.get("model")
    lines = [f"profile: {name}  ({record.get('model_id', '')})"]
    m = record.get("machine", {})
    lines.append(f"machine: {m.get('gpu', '?')} / {m.get('arch', '?')} / {m.get('backend', '?')}")
    lines.append(
        "note: module self% = fraction of phase wall time. Large self% on container/Embedding"
        " classes means CPU launch-gap (decode is launch-bound) — the ops lens is the"
        " authoritative GPU-time breakdown."
    )
    for phase, data in record.get("phases", {}).items():
        lines.append(f"\n=== phase: {phase} ===")
        lines.append("[modules — self time by class]")
        lines.append(_module_table(data.get("modules", [])))
        lines.append("[ops — top CUDA kernels (torch.profiler)]")
        lines.append(_op_table(data.get("ops", [])))
    inductor = record.get("inductor", [])
    nkern = sum(len(e.get("kernels", [])) for e in inductor)
    lines.append(
        f"\n=== inductor ===\n  {len(inductor)} graph region(s), {nkern} fused triton kernels (see inductor/INDEX.md)"
    )
    for e in inductor:
        ks = ", ".join(k.replace("triton_", "") for k in e.get("kernels", [])[:6])
        lines.append(f"  {e['region']}: {ks}{' ...' if len(e.get('kernels', [])) > 6 else ''}")
    return "\n".join(lines)

"""Measurement: capture run output and reduce timings to stats."""

import contextlib
import io
import sys

from kernels.cli.benchmark import TimingResults, _calculate_iqr_and_outliers


@contextlib.contextmanager
def capture():
    """Redirect stderr into a buffer for logging, then echo it back to the console."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield buf
    sys.stderr.write(buf.getvalue())


def stats(times, verified, ref_ms):
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    q1, q3, iqr, outliers = _calculate_iqr_and_outliers(times)
    return TimingResults(
        mean_ms=round(mean, 4),
        std_ms=round(var**0.5, 4),
        min_ms=round(min(times), 4),
        max_ms=round(max(times), 4),
        iterations=len(times),
        q1_ms=round(q1, 4),
        q3_ms=round(q3, 4),
        iqr_ms=round(iqr, 4),
        outliers=outliers,
        verified=verified,
        ref_mean_ms=ref_ms,
    )

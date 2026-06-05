import numpy as np


def clamp_bounds(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    scale: float,
) -> tuple[int, int, int, int]:
    return (
        int(np.floor(xmin / scale) * scale),
        int(np.floor(ymin / scale) * scale),
        int(np.ceil(xmax / scale) * scale),
        int(np.ceil(ymax / scale) * scale),
    )

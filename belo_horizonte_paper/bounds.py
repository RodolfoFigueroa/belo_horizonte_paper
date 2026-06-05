import os
from pathlib import Path
from typing import Literal, overload

import ee
import geemap
import geopandas as gpd


@overload
def load_bounds(
    data_path: os.PathLike,
    *,
    return_geometry: Literal[False] = False,
) -> ee.Geometry: ...


@overload
def load_bounds(
    data_path: os.PathLike,
    *,
    return_geometry: Literal[True] = True,
) -> tuple[ee.Geometry, gpd.GeoDataFrame]: ...


def load_bounds(
    data_path: os.PathLike,
    *,
    return_geometry: bool = False,
) -> ee.Geometry | tuple[ee.Geometry, gpd.GeoDataFrame]:
    data_path = Path(data_path)

    df_bounds = (
        gpd.read_file(
            data_path / "MANCHA_URBANA_2018.zip",
        )
        .pipe(lambda df: df.to_crs(df.estimate_utm_crs()))
        .explode()
        .assign(area=lambda df: df["geometry"].area)
        .sort_values("area", ascending=False)
        .head(1)
        .assign(geometry=lambda df: df["geometry"].simplify(10))
        .filter(["geometry"])
        .to_crs("EPSG:4326")
    )

    bounds_ee = geemap.geopandas_to_ee(df_bounds).first().geometry()
    if return_geometry:
        return bounds_ee, df_bounds
    return bounds_ee

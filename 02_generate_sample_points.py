import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells

    import os
    import warnings
    from pathlib import Path
    from typing import Literal

    import ee
    import geemap
    import geopandas as gpd
    import numpy as np
    import osmnx as ox
    import pandas as pd
    import rioxarray as rxr
    import shapely
    import spreg
    import statsmodels.formula.api as smf
    import xarray as xr
    from esda.moran import Moran
    from libpysal.weights import KNN, DistanceBand, W
    from xrspatial import aspect, slope

    from belo_horizonte_paper.bounds import load_bounds
    from belo_horizonte_paper.constants import LST_DEFAULT_YEAR
    from belo_horizonte_paper.temperature import get_lst
    from belo_horizonte_paper.utils import clamp_bounds

    ee.Initialize()


@app.cell
def _():
    data_path = Path(os.environ["DATA_PATH"])
    generated_path = data_path / "generated"
    return data_path, generated_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bounds
    """)
    return


@app.cell
def _(data_path):
    bounds_ee, bounds = load_bounds(data_path=data_path, return_geometry=True)
    return bounds, bounds_ee


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LST
    """)
    return


@app.cell
def _(bounds_ee):
    img_lst = get_lst(bounds_ee)
    lst_proj = img_lst.projection()
    crs = str(lst_proj.crs().getInfo())
    return crs, img_lst


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zones
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Points
    """)
    return


@app.cell
def _(crs, data_path):
    df_zones = (
        gpd.read_file(data_path / "PL_limite_area_verde_publica.zip")
        .drop(columns=["fid"])
        .assign(ID_AREA_VE=lambda df: df["ID_AREA_VE"].astype(int))
        .set_index("ID_AREA_VE")[["geometry"]]
        .sort_index()
        .to_crs(crs)
        .explode()
        .reset_index(names="orig_zone_id")
        .assign(
            dup_idx=lambda df: df.groupby("orig_zone_id").cumcount(),
            orig_zone_id=lambda df: (
                df["orig_zone_id"].astype(str).str.cat(df["dup_idx"].astype(str), sep="_")
            ),
        )
        .drop(columns=["dup_idx"])
        .set_index("orig_zone_id")
    )

    df_zones_joined = gpd.GeoDataFrame(
        geometry=list(df_zones.buffer(50).union_all().geoms),  # ty:ignore[unresolved-attribute]
        crs=crs,
    ).reset_index(names="zone_id")

    df_zones = df_zones.assign(
        zone_id=lambda df: df.sjoin(df_zones_joined, how="left", predicate="intersects")[  # ty:ignore[call-non-callable]
            "zone_id"
        ]
    )

    df_zones_joined = df_zones_joined.merge(
        df_zones[["zone_id", "geometry"]]
        .dissolve("zone_id")
        .reset_index()
        .rename(columns={"geometry": "geometry_unbuffered"}),
        on="zone_id",
        how="inner",
    )
    return (df_zones_joined,)


@app.cell
def _(crs):
    def get_zone_sample_points(zone: pd.Series) -> gpd.GeoDataFrame:
        zone_bounds = clamp_bounds(*zone["geometry"].bounds, scale=30)
        zone_points = gpd.GeoDataFrame(
            geometry=[
                shapely.Point(x, y)
                for x in np.arange(zone_bounds[0], zone_bounds[2], 30)
                for y in np.arange(zone_bounds[1], zone_bounds[3], 30)
            ],
            crs=crs,
        )
        return zone_points[zone_points.intersects(zone["geometry"])].assign(
            zone_id=zone["zone_id"],
        )

    return (get_zone_sample_points,)


@app.cell
def _(crs, df_zones_joined, generated_path, get_zone_sample_points):
    df_sample_points = pd.concat(
        [get_zone_sample_points(zone) for _, zone in df_zones_joined.iterrows()],
        ignore_index=True,
    ).pipe(lambda df: gpd.GeoDataFrame(df, geometry="geometry", crs=crs))

    col_sample_points: ee.FeatureCollection = geemap.geopandas_to_ee(
        df_sample_points.to_crs("EPSG:4326")
    )

    col_sample_points_buffered = col_sample_points.map(lambda feature: feature.buffer(50))

    df_sample_points.to_file(generated_path / "sample_points.gpkg")
    return col_sample_points, col_sample_points_buffered, df_sample_points


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data
    """)
    return


@app.cell
def _(crs):
    def sample_image(
        image: ee.Image,
        col: ee.FeatureCollection,
        reducer: ee.Reducer | None = None,
        *,
        scale: float,
    ) -> pd.Series:
        if reducer is None:
            reducer = ee.Reducer.first()

        reducer = reducer.setOutputs(["reduced"])

        return ee.data.computeFeatures(
            {
                "expression": image.reduceRegions(
                    collection=col,
                    reducer=reducer,
                    crs=crs,
                    scale=scale,
                    tileScale=4,
                ),
                "fileFormat": "GEOPANDAS_GEODATAFRAME",
            }
        )["reduced"]

    return (sample_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Canopy
    """)
    return


@app.cell
def _(bounds_ee, col_sample_points: ee.FeatureCollection, sample_image):
    img_canopy = (
        ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height")
        .filterBounds(bounds_ee)
        .first()
        .gte(ee.Number(3))
    )

    img_canopy_fraction = img_canopy.reduceResolution(
        reducer=ee.Reducer.mean(),
        bestEffort=True,
        maxPixels=1024,
    )

    sampled_trees = sample_image(img_canopy_fraction, col_sample_points, scale=30).rename(
        "trees"
    )
    return (sampled_trees,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lights
    """)
    return


@app.cell
def _(bounds_ee, col_sample_points: ee.FeatureCollection, sample_image):
    img_lights = (
        ee.imagecollection.ImageCollection(
            "projects/sat-io/open-datasets/srunet-npp-viirs-ntl",
        )
        .filterBounds(bounds_ee)
        .filter(ee.filter.Filter.eq("id_no", "SRUNet_NPP_VIIRS_V2_Like_2020"))
        .first()
    )

    sampled_lights = sample_image(
        img_lights, col_sample_points, scale=img_lights.projection().nominalScale()
    ).rename("lights")
    return (sampled_lights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LST
    """)
    return


@app.cell
def _(col_sample_points: ee.FeatureCollection, img_lst, sample_image):
    sampled_lst = sample_image(img_lst, col_sample_points, scale=30).rename("lst")
    return (sampled_lst,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Topography
    """)
    return


@app.cell
def _(df_sample_points):
    def sample_arr(arr: xr.DataArray, points: gpd.GeoDataFrame) -> pd.Series:
        temp = df_sample_points.to_crs(arr.rio.crs)
        return pd.Series(
            arr.sel(
                x=xr.DataArray(temp.geometry.x),
                y=xr.DataArray(temp.geometry.y),
                method="nearest",
            ).values.flatten(),
            index=points.index,
        )

    return (sample_arr,)


@app.cell
def _(generated_path):
    arr_elevation = xr.DataArray(
        rxr.open_rasterio(generated_path / "elevation.tif")
    ).squeeze("band", drop=True)

    arr_slope = slope(arr_elevation)
    arr_aspect = aspect(arr_elevation)
    return arr_aspect, arr_elevation, arr_slope


@app.cell
def _(arr_aspect, arr_elevation, arr_slope, df_sample_points, sample_arr):
    sampled_elevation = sample_arr(arr_elevation, df_sample_points).rename("elevation")

    sampled_slope = sample_arr(
        arr_slope,
        df_sample_points,
    ).rename("slope")

    sampled_aspect = sample_arr(arr_aspect, df_sample_points).rename("aspect")
    return sampled_aspect, sampled_elevation, sampled_slope


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Land use
    """)
    return


@app.cell
def _(crs, data_path, df_sample_points):
    df_lots = (
        gpd.read_file(
            data_path / "TIPOLOGIA_USO_OCUPACAO_LOTE_2022.zip",
        )
        .to_crs(crs)
        .query("TIPOLOGIA_ != 'SEM INFORMACAO'")
    )

    df_sampled_land_use = (
        df_sample_points[["geometry"]]
        .assign(
            geometry=lambda df: df["geometry"].buffer(50),
            orig_area=lambda df: df["geometry"].area,
        )
        .reset_index(names="index")
        .overlay(df_lots[["TIPOLOGIA_", "geometry"]], how="intersection")
        .assign(area_frac=lambda df: df["geometry"].area / df["orig_area"])
        .groupby(["index", "TIPOLOGIA_"])["area_frac"]
        .sum()
        .unstack()
        .fillna(0)
        .pipe(lambda df: df.div(df.sum(axis=1), axis=0))
        .rename(
            columns={
                "LOTE VAGO": "landuse_vacant_lot",
                "MISTO": "landuse_mixed_use",
                "NAO RESIDENCIAL": "landuse_non_residential",
                "PARQUE": "landuse_park",
                "RESIDENCIAL": "landuse_residential",
            },
        )
        .fillna(0)
    )
    return (df_sampled_land_use,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distance to edge
    """)
    return


@app.cell
def _(df_sample_points, df_zones_joined):
    sampled_distances_edge = (
        df_sample_points.rename(columns={"geometry": "point"})
        .merge(
            df_zones_joined.rename(
                columns={"geometry": "zone", "geometry_unbuffered": "zone_unbuffered"}
            ),
            on="zone_id",
            how="inner",
        )
        .assign(
            zone_boundary=lambda df: df["zone_unbuffered"].boundary,
            is_inside=lambda df: df["zone_unbuffered"].contains(df["point"]),
            boundary_dist=lambda df: df["zone_boundary"].distance(df["point"]),
            dist_inside=lambda df: df["boundary_dist"].where(df["is_inside"], 0),
            dist_outside=lambda df: df["boundary_dist"].where(~df["is_inside"], 0),
        )
        .assign(is_inside=lambda df: df["is_inside"].astype(int))
        .assign(
            dist_sign=lambda df: df["is_inside"].astype(int).mul(2).add(-1),
            signed_distance=lambda df: df["dist_sign"].mul(df["boundary_dist"]),
        )
        .drop(
            columns=[
                "point",
                "zone",
                "zone_unbuffered",
                "zone_boundary",
                "boundary_dist",
                "dist_sign",
                "zone_id",
            ]
        )
    )
    return (sampled_distances_edge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impervious surface
    """)
    return


@app.cell
def _(col_sample_points: ee.FeatureCollection, sample_image):
    img_impervious = ee.Image("projects/sat-io/open-datasets/GISA_1972_2021")
    img_impervious = img_impervious.gt(ee.Number(0)).And(img_impervious.lt(ee.Number(38)))

    sampled_impervious_binary = sample_image(
        img_impervious, col_sample_points, scale=30
    ).rename("is_impervious")
    return img_impervious, sampled_impervious_binary


@app.cell
def _(col_sample_points_buffered, img_impervious, sample_image):
    sampled_impervious = sample_image(
        img_impervious, col_sample_points_buffered, reducer=ee.Reducer.mean(), scale=30
    ).rename("impervious_frac")
    return (sampled_impervious,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building height
    """)
    return


@app.cell
def _(bounds_ee, col_sample_points_buffered, sample_image):
    building_height_year = LST_DEFAULT_YEAR + 1

    img_height = (
        ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
        .filterBounds(bounds_ee)
        .filterDate(f"{building_height_year}-01-01", f"{building_height_year}-12-31")
        .first()
        .select("building_height")
    )

    sampled_height = sample_image(
        img_height, col_sample_points_buffered, ee.Reducer.mean(), scale=4
    ).rename("building_height")
    return (sampled_height,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Water fraction
    """)
    return


@app.cell
def _(bounds_ee, col_sample_points_buffered, sample_image):
    img_class = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(bounds_ee).first()

    img_water = (
        img_class.eq(80)  # Permanent water bodies
        .Or(img_class.eq(90))  # Herbaceous wetland
        .Or(img_class.eq(95))  # Mangroves
        .multiply(ee.Image.pixelArea())
    )


    sampled_water = sample_image(
        img_water, col_sample_points_buffered, reducer=ee.Reducer.sum(), scale=10
    )
    return


@app.cell
def _(bounds):
    features_water = ox.features_from_bbox(
        tuple(bounds.total_bounds), tags={"water": True, "waterway": True}
    )
    features_water = (
        features_water.assign(
            water_type=lambda df: pd.concat(
                [df["water"].dropna(), df["waterway"].dropna()]
            ),
            intermittent=lambda df: (
                df["intermittent"].fillna("no").map({"yes": True, "no": False}).astype(bool)
            ),
            tunnel=lambda df: (
                df["tunnel"].where(df["tunnel"].isna(), True).fillna(False).astype(bool)
            ),
        )
        .drop(columns=["water", "waterway"])
        .loc[
            lambda df: (
                (df["geometry"].geom_type != "Point")
                & (
                    ~df["water_type"].isin(["dam", "pool", "reflecting_pool"])
                )  # Remove extraneous types
                & (~df["intermittent"])  # Remove intermittent sources
                & (~df["tunnel"])  # Remove underground streams
            )
        ]
    )
    return (features_water,)


@app.cell
def _(df_sample_points, features_water):
    natural_water_types = ["stream", "river", "lake", "reservoir", "pond"]
    standing_water_types = ["lake", "reservoir", "pond"]

    sampled_water_dist = (
        df_sample_points.sjoin_nearest(
            features_water.loc[
                lambda df: df["water_type"].isin(natural_water_types), ["geometry"]
            ].to_crs(df_sample_points.crs),
            distance_col="dist",
        )
        .sort_values("dist", ascending=True)
        .reset_index(names="index")
        .drop_duplicates(subset=["index"])
        .set_index("index")
        ["dist"]
        .rename("dist_to_water")
    )

    sampled_is_near_standing_water = (
        df_sample_points.sjoin_nearest(
            features_water.loc[
                lambda df: df["water_type"].isin(standing_water_types), ["geometry"]
            ].to_crs(df_sample_points.crs),
            distance_col="dist",
            max_distance=250,
        )
        .assign(dist=True)
        ["dist"]
        .reindex(df_sample_points.index, fill_value=False)
        .astype(bool)
        .astype(int)
        .rename("is_near_standing_water")
    )
    return sampled_is_near_standing_water, sampled_water_dist


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final
    """)
    return


@app.cell
def _(
    crs,
    df_sample_points,
    df_sampled_land_use,
    sampled_aspect,
    sampled_distances_edge,
    sampled_elevation,
    sampled_height,
    sampled_impervious,
    sampled_impervious_binary,
    sampled_is_near_standing_water,
    sampled_lights,
    sampled_lst,
    sampled_slope,
    sampled_trees,
    sampled_water_dist,
):
    df_model = pd.concat(
        [
            sampled_trees,
            sampled_lst,
            sampled_elevation,
            sampled_slope,
            sampled_aspect,
            sampled_lights,
            sampled_impervious_binary,
            sampled_impervious,
            sampled_height,
            sampled_water_dist,
            sampled_is_near_standing_water,
        ],
        axis=1,
    ).assign(elevation=sampled_elevation)

    df_model = (
        pd.concat(
            [df_model, df_sampled_land_use, df_sample_points, sampled_distances_edge],
            axis=1,
        )
        .assign(
            lights_log=lambda df: np.log(df["lights"]),
            aspect_rad=lambda df: np.deg2rad(df["aspect"]),
            aspect_sin=lambda df: np.sin(df["aspect_rad"]),
            aspect_cos=lambda df: np.cos(df["aspect_rad"]),
        )
        .pipe(lambda df: gpd.GeoDataFrame(df, geometry="geometry", crs=crs))
    )
    return (df_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OLS
    """)
    return


@app.cell
def _(crs, df_model):
    vars_ = [
        "trees",
        "elevation",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "lights_log",
        "landuse_mixed_use",
        "landuse_non_residential",
        "landuse_park",
        "landuse_residential",
        "is_near_standing_water",
        "dist_to_water",
        # "is_inside",
        "dist_inside",
        "dist_outside",
        # "signed_distance",
        "impervious_frac",
        "building_height",
    ]

    df_analysis = gpd.GeoDataFrame(
        df_model.dropna(subset=["lst", *vars_]).copy(),
        geometry="geometry",
        crs=crs,
    )
    return df_analysis, vars_


@app.cell
def _(df_analysis, vars_):
    ols = smf.ols(
        "lst ~ " + " + ".join(vars_),
        data=df_analysis,
    ).fit()
    print(ols.summary())
    return (ols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spatial
    """)
    return


@app.cell
def _(df_analysis):
    def build_within_zone_weights(
        gdf: gpd.GeoDataFrame,
        *,
        zone_col: str = "zone_id",
        method: Literal["knn", "distance"],
        param: float = 8,
        suppress_warnings: bool = True,
    ) -> W:
        warning_level = "ignore" if suppress_warnings else "default"

        neighbors = {}
        weights = {}

        for _, zone_points in gdf.groupby(zone_col, sort=False):
            zone_ids = zone_points.index.to_list()

            if len(zone_ids) == 1:
                neighbors[zone_ids[0]] = []
                weights[zone_ids[0]] = []
                continue

            with warnings.catch_warnings():
                warnings.simplefilter(warning_level, category=UserWarning)
                if method == "knn":
                    zone_w = KNN.from_dataframe(
                        zone_points,
                        k=min(int(param), len(zone_ids) - 1),
                        ids=zone_ids,
                    )
                else:
                    zone_w = DistanceBand.from_dataframe(
                        zone_points,
                        threshold=param,
                        ids=zone_ids,
                    )
                neighbors.update(zone_w.neighbors)
                weights.update(zone_w.weights)

        with warnings.catch_warnings():
            warnings.simplefilter(warning_level, category=UserWarning)
            w = W(neighbors, weights, ids=gdf.index.to_list())
            w.transform = "r"
            return w


    w = build_within_zone_weights(
        df_analysis, zone_col="zone_id", method="knn", param=8, suppress_warnings=True
    )
    return build_within_zone_weights, w


@app.cell
def _(ols, w):
    moran_resid = Moran(ols.resid.to_numpy(), w)
    print("Moran's I:", moran_resid.I)
    print("p-value:", moran_resid.p_sim)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OLS
    """)
    return


@app.cell
def _(df_analysis, vars_, w):
    y = df_analysis["lst"].to_numpy().reshape(-1, 1)
    X = df_analysis[vars_].to_numpy()

    ols_sp = spreg.OLS(
        y,
        X,
        w=w,
        spat_diag=True,
        moran=True,
        name_y="lst",
        name_x=vars_,
    )

    print(ols_sp.summary)
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SEM
    """)
    return


@app.cell
def _(X, vars_, w, y):
    err_model = spreg.ML_Error(
        y,
        X,
        w=w,
        name_y="lst",
        name_x=vars_,
    )
    print(err_model.summary)
    return (err_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SLM
    """)
    return


@app.cell
def _(X, vars_, w, y):
    lag_model = spreg.ML_Lag(
        y,
        X,
        w=w,
        name_y="lst",
        name_x=vars_,
    )
    print(lag_model.summary)
    return (lag_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SDM
    """)
    return


@app.cell
def _(X, vars_, w, y):
    sdm_model = spreg.ML_Lag(
        y,
        X,
        w=w,
        slx_lags=1,
        slx_vars="All",
        name_y="lst",
        name_x=vars_,
        spat_impacts="full",
    )

    print(sdm_model.summary)
    return (sdm_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _(err_model, lag_model, sdm_model, w):
    def summarize_spatial_model(model, w, name):
        moran = Moran(model.u.ravel(), w)
        return {
            "model": name,
            "logll": model.logll,
            "aic": model.aic,
            "bic": model.schwarz,
            "resid_moran_I": moran.I,
            "resid_moran_p": moran.p_sim,
        }


    comparison = pd.DataFrame(
        [
            summarize_spatial_model(err_model, w, "ML_Error"),
            summarize_spatial_model(lag_model, w, "ML_Lag"),
            summarize_spatial_model(sdm_model, w, "SDM"),
        ]
    )

    comparison.sort_values("aic")
    return


@app.cell
def _(X, build_within_zone_weights, df_analysis, vars_, y):
    results = []

    for method, params in zip(
        ("knn", "distance"), [list(range(4, 12, 2)), list(range(50, 110, 10))]
    ):
        for param in params:
            w_k = build_within_zone_weights(
                df_analysis, zone_col="zone_id", method=method, param=param
            )

            model = spreg.ML_Lag(
                y,
                X,
                w=w_k,
                slx_lags=1,
                slx_vars="All",
                name_y="lst",
                name_x=vars_,
                spat_impacts="full",
            )

            moran = Moran(model.u.ravel(), w_k)

            results.append(
                {
                    "method": method,
                    "param": param,
                    "logll": model.logll,
                    "aic": model.aic,
                    "bic": model.schwarz,
                    "resid_moran_I": moran.I,
                    "resid_moran_p": moran.p_sim,
                }
            )

    full_comparison = pd.DataFrame(results).sort_values("aic")
    full_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SDM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distance band
    """)
    return


@app.cell
def _(X, build_within_zone_weights, df_analysis, vars_, y):
    w_dist_band = build_within_zone_weights(
        df_analysis, zone_col="zone_id", method="distance", param=50
    )

    sdm_second = spreg.ML_Lag(
        y,
        X,
        w=w_dist_band,
        slx_lags=1,
        slx_vars="All",
        name_y="lst",
        name_x=vars_,
        spat_impacts="full",
    )

    print(sdm_second.summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KNN
    """)
    return


@app.cell
def _(X, build_within_zone_weights, df_analysis, vars_, y):
    w_knn = build_within_zone_weights(
        df_analysis, zone_col="zone_id", method="knn", param=4
    )

    knn_second = spreg.ML_Lag(
        y,
        X,
        w=w_knn,
        slx_lags=1,
        slx_vars="All",
        name_y="lst",
        name_x=vars_,
        spat_impacts="full",
    )

    print(knn_second.summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Final model
    """)
    return


@app.cell
def _(build_within_zone_weights, df_analysis, y):
    final_vars = [
        "trees",
        "elevation",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "lights_log",
        "landuse_non_residential",
        "landuse_residential",
        "dist_outside",
        "impervious_frac",
        "building_height",
    ]

    X_final = df_analysis[final_vars].to_numpy()

    w_final = build_within_zone_weights(
        df_analysis, zone_col="zone_id", method="distance", param=50
    )

    model_final = spreg.ML_Lag(
        y,
        X_final,
        w=w_final,
        slx_lags=1,
        slx_vars="All",
        name_y="lst",
        name_x=final_vars,
        spat_impacts="full",
    )

    print(model_final.summary)
    return


if __name__ == "__main__":
    app.run()

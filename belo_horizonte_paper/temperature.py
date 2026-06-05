import ee

from belo_horizonte_paper.constants import (
    LST_DEFAULT_COL,
    LST_DEFAULT_QA_BAND,
    LST_DEFAULT_TEMP_BAND,
    LST_DEFAULT_YEAR,
)


def extract_bits(img: ee.Image, from_bit: int, to_bit: int) -> ee.Image:
    mask_size = ee.Number(1).add(to_bit).subtract(from_bit)
    mask = ee.Number(1).leftShift(mask_size).subtract(1)
    return img.rightShift(from_bit).bitwiseAnd(mask)


def mask_clouds(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    mask = extract_bits(qa, 1, 4).eq(0)
    return img.updateMask(mask)


def get_lst(
    bounds: ee.Geometry,
    *,
    col_name: str = LST_DEFAULT_COL,
    year: int = LST_DEFAULT_YEAR,
    temp_band: str = LST_DEFAULT_TEMP_BAND,
    qa_band: str = LST_DEFAULT_QA_BAND,
) -> ee.Image:
    col: ee.ImageCollection = (
        ee.ImageCollection(col_name)
        .filterBounds(bounds)
        .filterDate(f"{year}-12-21", f"{year + 1}-03-21")
        .select([temp_band, qa_band])
    )

    proj = col.first().projection()

    return (
        col.map(mask_clouds)
        .select(temp_band)
        .map(lambda img: img.multiply(0.00341802).add(149 - 273.15))
        .mean()
        .rename(f"b{year}")
        .reproject(crs=proj.crs(), scale=30)
    )

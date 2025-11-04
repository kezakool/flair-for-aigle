import os
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.mask import mask
from rasterio.transform import array_bounds
from typing import Dict
from shapely.geometry import Polygon
import logging
logger = logging.getLogger(__name__)

def create_box_from_bounds(x_min: float, x_max: float, y_min: float, y_max: float) -> box:
    """
    Create a shapely polygon from given bounding box coordinates.
    """
    return box(x_min, y_max, x_max, y_min)


def generate_patches_from_reference(config: Dict, img_path: str, geozone_contour_geometries: Polygon) -> gpd.GeoDataFrame:
    """
    Slice the reference raster into overlapping tiles based on patch size and margin.
    Args:
        config: Inference configuration dictionary
    Returns:
        GeoDataFrame with tile metadata and geometries
    """
    patch_size = config['img_pixels_detection']
    margin = config['margin']
    output_path = config['output_path']
    output_name = config['output_name']
    write_dataframe = config.get('write_dataframe', False)

    # Open the image corresponding to the reference modality
    ref_mod = config['reference_modality']

    with rasterio.open(img_path) as src:
        profile = src.profile
        src_height, src_width = src.shape[0], src.shape[1]
        # extract intersection of src image and geozone
        try:
            out_image, out_transform = mask(src, geozone_contour_geometries, crop=True)
        except ValueError:
            return gpd.GeoDataFrame()
        
        # Compute bounds from transform + array shape
        height, width = out_image.shape[1], out_image.shape[2]
        left_overall, bottom_overall, right_overall, top_overall = array_bounds(height, width, out_transform)
        ref_left_overall, ref_bottom_overall, ref_right_overall, ref_top_overall = array_bounds(src_height, src_width, src.transform)

    resolution = config['reference_resolution']

    # Convert patch and margin to physical size
    geo_output_size = (patch_size * resolution, patch_size * resolution)
    geo_margin = (margin * resolution, margin * resolution)

    # Compute stride (step between patch centers)
    geo_step = (
        (patch_size - 2 * margin) * resolution,
        (patch_size - 2 * margin) * resolution
    )

    min_x, min_y = left_overall, bottom_overall
    max_x, max_y = right_overall, top_overall

    tiles = []
    existing_patches = set()

    for x_coord in np.arange(min_x - geo_margin[0], max_x + geo_margin[0], geo_step[0]):
        for y_coord in np.arange(min_y - geo_margin[1], max_y + geo_margin[1], geo_step[1]):

            # Adjust patch bounds to remain within the image
            if x_coord + geo_output_size[0] > max_x + geo_margin[0]:
                x_coord = max_x + geo_margin[0] - geo_output_size[0]
            if y_coord + geo_output_size[1] > max_y + geo_margin[1]:
                y_coord = max_y + geo_margin[1] - geo_output_size[1]

            left = x_coord + geo_margin[0]
            right = min(x_coord + geo_output_size[0] - geo_margin[0], max_x)
            bottom = y_coord + geo_margin[1]
            top = min(y_coord + geo_output_size[1] - geo_margin[1], max_y)

            patch_bounds = tuple(round(val, 6) for val in (left, bottom, right, top))
            if patch_bounds in existing_patches:
                continue

            existing_patches.add(patch_bounds)
            
            col = int((x_coord - ref_left_overall) // resolution) + 1
            row = int((y_coord - ref_bottom_overall) // resolution) + 1

            if right-left>0 and top-bottom>0 :
                tiles.append({
                    "id": f"{1}-{row}-{col}",
                    "input_id": img_path,
                    "output_id": output_name,
                    "job_done": 0,
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "top": top,
                    "left_o": left_overall,
                    "bottom_o": bottom_overall,
                    "right_o": right_overall,
                    "top_o": top_overall,
                    "geometry": create_box_from_bounds(
                        x_coord,
                        x_coord + geo_output_size[0],
                        y_coord,
                        y_coord + geo_output_size[1]
                    )
                })

    gdf_output = gpd.GeoDataFrame(tiles, crs=profile['crs'], geometry='geometry')

    if write_dataframe:
        gpkg_path = os.path.join(output_path, output_name + '_slicing_job.gpkg')
        logger.info(f"[✓] Saved Sliced Boxes: {gpkg_path}")
        gdf_output.to_file(gpkg_path, driver='GPKG')

    return gdf_output

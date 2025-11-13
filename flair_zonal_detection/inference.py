import os
import sys
import torch
import rasterio
import time
import geopandas as gpd
import numpy as np
import logging
from itertools import islice
from typing import Dict, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
from rasterio.io import DatasetReader
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.transform import rowcol
from shapely.geometry import shape
from rasterio.transform import from_origin
from scipy.ndimage import zoom
from multiprocessing import Pool, cpu_count
from flair_zonal_detection.config import (
                                            load_config,
                                            validate_config,
                                            config_recap_1, config_recap_2
)
from flair_hub.utils.messaging import Logger
from flair_zonal_detection.dataset import MultiModalSlicedDataset
from flair_zonal_detection.postprocess import (
                                                convert,
                                                convert_to_cog
)
from flair_zonal_detection.model_utils import (
                                                build_inference_model,
                                                compute_patch_sizes
)
from flair_zonal_detection.slicing import generate_patches_from_reference


logger = logging.getLogger(__name__)

def overwrite_config(config: str, model_ckpt_path: str, model_threshold_filepath: str, result_folder: str, log_folder: str) -> Dict:
    """
    Overwrite flair config
    """
    config['model_weights'] = model_ckpt_path
    config['model_threshold_filepath'] = model_threshold_filepath
    config['output_path'] = result_folder
    config['log_folder'] = log_folder
    
    return config
    

def prep_config(config_path: str, model_ckpt_path: str, model_threshold_filepath: str, result_folder: str, log_folder: str) -> Dict:
    """
    Load and validate configuration, initialize logging and device.
    """
    config = load_config(config_path)
    
    config = overwrite_config(config, model_ckpt_path, model_threshold_filepath, result_folder, log_folder)
    
    validate_config(config)
    config_recap_1(config)
    config = initialize_geometry_and_resolutions(config)
    config_recap_2(config)

    config['device'] = torch.device("cuda" if config.get("use_gpu", torch.cuda.is_available()) else "cpu")
    config['output_type'] = config.get("output_type", "argmax")
    return config


def initialize_geometry_and_resolutions(config: Dict) -> Dict:
    """
    Determine bounding box and resolution consistency across all active modalities.
    Sets:
        - config['reference_resolution']
        - config['modality_resolutions']
        - config['image_bounds']
    Validates:
        - All bounds match (warn or fail if not)
    """
    modalities = config['modalities']
    active_mods = [mod for mod, is_active in modalities['inputs'].items() if is_active]

    resolutions = {}
    bounds = []

    for mod in active_mods:
        path = modalities[mod]['input_img_path']
        with rasterio.open(path) as src:
            resolutions[mod] = round(src.res[0], 5)
            bounds.append((mod, src.bounds))
            
            if 'image_shape_px' not in config:
                config['image_shape_px'] = {
                    'height': src.height,
                    'width': src.width
                }

    # Check bounds match
    ref_mod, ref_bounds = bounds[0]
    for mod, b in bounds[1:]:
        if not np.allclose(b, ref_bounds, atol=1e-2):
            raise ValueError(
                f"[✗] Bounds mismatch between '{ref_mod}' and '{mod}':\n"
                f"  {ref_mod}: {ref_bounds}\n"
                f"  {mod}: {b}"
            )

    # Choose reference modality based on coarsest resolution (largest m/px)
    ref_mod, reference_resolution = min(resolutions.items(), key=lambda x: x[1])
    config['reference_modality'] = ref_mod
    config['reference_resolution'] = reference_resolution

    config['modality_resolutions'] = resolutions
    config['image_bounds'] = {
        'left': ref_bounds.left,
        'bottom': ref_bounds.bottom,
        'right': ref_bounds.right,
        'top': ref_bounds.top
    }

    tile_size_m = config['img_pixels_detection'] * reference_resolution
    margin_size_m = config['margin'] * reference_resolution
    config['tile_size_m'] = round(tile_size_m, 2)
    config['margin_size_m'] = round(margin_size_m, 2)

    return config



def prep_dataset(config: Dict, tiles_gdf, patch_sizes: Dict[str, int]) -> MultiModalSlicedDataset:
    """
    Prepare the dataset object from config and sliced patches.
    """
    active_mods = [m for m, active in config['modalities']['inputs'].items() if active]
    modality_cfgs = {m: config['modalities'][m] for m in active_mods}

    config['labels'] = [t['name'] for t in config['tasks'] if t['active']]
    config['labels_configs'] = {
        t['name']: {'value_name': t['class_names']} for t in config['tasks'] if t['active']
    }

    return MultiModalSlicedDataset(
        dataframe=tiles_gdf,
        modality_cfgs=modality_cfgs,
        patch_size_dict=patch_sizes,
        ref_date_str=config['multitemp_model_ref_date'],
        modalities_config=config
    )


def init_outputs(config: Dict, ref_img: DatasetReader, i) -> Tuple[Dict[str, DatasetReader], Dict[str, str]]:
    """
    Initialize output raster files per task. Adjusts dimensions and transform if resolution differs.
    """
    output_files = {}
    temp_paths = {}
    output_type = config['output_type']
    ref_res = config['reference_resolution']
    out_res = config.get("output_px_meters", ref_res)
    image_bounds = config['image_bounds']
    needs_rescale = abs(ref_res - out_res) > 1e-6

    for task in config['tasks']:
        if not task['active']:
            continue

        num_classes = len(task['class_names'])
        suffix = 'argmax' if output_type == 'argmax' else 'class-prob'
        out_path = os.path.join(
            config['output_path'],
            f"{config['output_name']}_{task['name']}_{suffix}_i.tif"
        )

        if not needs_rescale:
            # Use reference image profile directly
            profile = ref_img.profile.copy()
            profile.update({
                "count": num_classes if output_type == "class_prob" else 1,
                "dtype": "uint8",
                "compress": "lzw"
            })
        else:
            # Adjust height, width and transform based on new resolution
            out_height = int(round((image_bounds['top'] - image_bounds['bottom']) / out_res))
            out_width = int(round((image_bounds['right'] - image_bounds['left']) / out_res))
            transform = from_origin(image_bounds['left'], image_bounds['top'], out_res, out_res)

            profile = {
                "driver": "GTiff",
                "height": out_height,
                "width": out_width,
                "count": num_classes if output_type == "class_prob" else 1,
                "dtype": "uint8",
                "crs": ref_img.crs,
                "transform": transform,
                "compress": "lzw"
            }

        output_files[task['name']] = rasterio.open(out_path, 'w', **profile)
        temp_paths[task['name']] = out_path

    return output_files, temp_paths



def resample_prediction(prediction: np.ndarray, scale: float) -> np.ndarray:
    """
    Resample prediction using nearest-neighbor zoom.
    
    Handles both:
    - (H, W) for argmax
    - (C, H, W) for logits/class-probabilities
    """
    if prediction.ndim == 2: 
        return zoom(prediction, zoom=scale, order=0)
    elif prediction.ndim == 3:
        c, h, w = prediction.shape
        return zoom(prediction, zoom=(1, scale, scale), order=0)
    else:
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")


def load_geozone_contour(config):
    """
       Load geometry of processed geozone 
    """
    geozones_shapefilename = os.path.join(config.db_sources, os.getenv('GEOZONES_SHAPEFILE'))
    if not os.path.exists(geozones_shapefilename):
        logger.warning(f"Geozones shapefile not found, expecting cache file : {geozones_shapefilename}")
        logger.info(f"Querying geozones from aigle bd topo...")
        try:
            # Read the SQL query into a GeoDataFrame
            gdf = gpd.read_postgis("""select id, "name", geometry, geo_zone_type, name_normalized, iso_code  from detections.fr_geozone_view""", con=os.getenv('DB_STRING_PROD'), geom_col='geometry')  # Ensure 'geometry' is your geometry column

            # Save the GeoDataFrame to a shapefile
            gdf.to_file(geozones_shapefilename)
            logger.info(f"Shapefile geozones created successfully at { geozones_shapefilename}")
        except Exception as e :
            logger.critical(f"A critical error occurred during query : {e}", exc_info=True)
            sys.exit(1)
    
    gdf_geozone = gpd.read_file(geozones_shapefilename)
    gdf_geozone.to_crs(config.input_crs, inplace=True)
    geozone_contour_geometries = gdf_geozone[gdf_geozone.iso_code==config.geozones_codes].geometry.values
    
    return geozone_contour_geometries

def inference_and_write(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tiles_gdf,
    config: Dict,
    output_files: Dict[str, DatasetReader],
    ref_img: DatasetReader
) -> None:
    """
    Run model inference and write predictions to raster files.
    Supports resampling logits to output_px_meters if different from reference_resolution.
    """
    device = config['device']
    margin_px = config['margin']
    tile_size = config['img_pixels_detection']
    output_type = config['output_type']
    ref_res = config['reference_resolution']
    out_res = config.get('output_px_meters', ref_res)  # fallback to ref_res if not set
    needs_rescale = abs(ref_res - out_res) > 1e-6
    image_left, image_bottom, image_right, image_top= list(ref_img.bounds)
    image_bounds = {'left': image_left, 'bottom': image_bottom, 'right': image_right, 'top': image_top}

    logger.info("\n[ ] Starting inference and writing raster tiles...\n")

    for batch in tqdm(dataloader, file=sys.stdout):
        #logger.info(f"batch num : {batch['index']}")
        inputs = {
            mod: batch[mod].to(device)
            for mod in batch if mod not in ['index'] and not mod.endswith('_DATES')
        }
        for mod in batch:
            if mod.endswith('_DATES'):
                inputs[mod] = batch[mod].to(device)

        indices = batch['index'].cpu().numpy().flatten()
        rows = tiles_gdf.iloc[indices]

        with torch.no_grad():
            logits_tasks, _ = model(inputs)

        for task_name, logits in logits_tasks.items():
            logits = logits.cpu().numpy()

            for i, idx in enumerate(indices):
                row = rows.iloc[i]

                logit_patch = logits[i, :, margin_px:tile_size - margin_px, margin_px:tile_size - margin_px]
                res_ref = config["reference_resolution"]
                res_out = config.get("output_px_meters", res_ref)
                needs_rescale = abs(res_out - res_ref) > 1e-6
                scale = res_ref / res_out if needs_rescale else 1.0

                #logger.info(f"Logits output desc : {logit_patch[0][0][:15]}")

                if output_type == "argmax":
                    prediction = convert(logit_patch, "argmax")  # shape: (H, W)
                    if needs_rescale:
                        prediction = resample_prediction(prediction, scale)
                else:
                    if needs_rescale:
                        logit_patch = resample_prediction(logit_patch, scale)
                    prediction = convert(logit_patch, output_type)  # (C, H, W)

                # Get top-left corner in output raster
                left = row['left']
                top = row['top']
                left_px = int(round((left - image_bounds['left']) / out_res))
                top_px  = int(round((image_bounds['top'] - top) / out_res))

                # Get prediction size
                height_px = prediction.shape[-2]
                width_px  = prediction.shape[-1]

                # Output raster dimensions
                img_height = int(round((image_bounds['top'] - image_bounds['bottom']) / out_res))
                img_width  = int(round((image_bounds['right'] - image_bounds['left']) / out_res))

                # Clip
                if top_px + height_px > img_height:
                    height_px = img_height - top_px
                if left_px + width_px > img_width:
                    width_px = img_width - left_px

                if height_px <= 0 or width_px <= 0:
                    logger.info(f"[!] Skipping tile {row['id']} — window out of bounds.")
                    continue

                # Crop prediction if needed
                prediction = prediction[..., :height_px, :width_px]
                window = Window(col_off=left_px, row_off=top_px, width=width_px, height=height_px,)
                
                #logger.info(f"Transfering prediction of tile {row['id']} into window {height_px}x{width_px} at position {left_px}-{top_px}")
                
                # Write
                if output_type == "argmax":
                    output_files[task_name].write(prediction[0], 1, window=window)
                else:
                    for c in range(prediction.shape[0]):
                        output_files[task_name].write(prediction[c], c + 1, window=window)

    for dst in output_files.values():
        dst.close()



def _extract_polygons_for_class(args):
    """Worker function to extract polygons for a given class."""
    cls, data, transform, min_area, simplification = args
    mask = data == cls
    if not np.any(mask):
        return []

    local_polygons = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        poly = shape(geom)
        if poly.is_empty or poly.area < min_area:
            continue
        if simplification > 0:
            poly = poly.simplify(simplification, preserve_topology=True)
        local_polygons.append({"class_id": int(cls), "geometry": poly})

    return local_polygons

def raster_to_polygons(
    tiff_path: str,
    ignore_background: bool = True,
    background_value: int = 18,
    min_area: float = 1.0,
    simplification: float = 0.1,
    n_jobs: int = None,
) -> gpd.GeoDataFrame:
    """
    Parallel extraction of vector polygons from a raster TIFF
    where integer values represent classes.
    """

    # --- Load raster ---
    with rasterio.open(tiff_path['AERIAL_LABEL-COSIA'].name) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    classes = np.unique(data)
    if ignore_background:
        classes = classes[classes != background_value]

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    # --- Parallel process ---
    args = [(cls, data, transform, min_area, simplification) for cls in classes]

    polygons = []
    with Pool(processes=n_jobs) as pool:
        for result in tqdm(pool.imap_unordered(_extract_polygons_for_class, args),
                           total=len(args), desc="Extracting polygons"):
            polygons.extend(result)

    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    return gdf

def raster_to_polygons_old(
    tiff_path: str,
    ignore_background: bool = True,
    background_value: int = 18,
    min_area: float = 1.0,                     # filter small polygons (sq meters)
    simplification: float = 0.1,               # polygon simplification tolerance
) -> gpd.GeoDataFrame:
    """
    Extracts vector polygons from a raster TIFF where integer values represent classes.
    
    Returns GeoDataFrame with: class_id, geometry
    """

    # Load raster
    with rasterio.open(tiff_path['AERIAL_LABEL-COSIA'].name) as src:
        data = src.read(1)  # first band
        transform = src.transform
        crs = src.crs

    # Unique class values
    classes = np.unique(data)

    if ignore_background:
        classes = classes[classes != background_value]

    polygons = []

    # Loop class by class for efficiency
    for cls in tqdm(classes, desc="Extracting polygons per class"):
        mask = data == cls
        if not np.any(mask):
            continue

        # shapes() extracts polygon boundaries for regions where mask==1
        for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
            poly = shape(geom)
            if poly.is_empty:
                continue
            if poly.area < min_area:
                continue

            if simplification > 0:
                poly = poly.simplify(simplification, preserve_topology=True)

            polygons.append({
                "class_id": int(cls),
                "geometry": poly
            })

    gdf = gpd.GeoDataFrame(polygons, crs=crs)

    return gdf

def inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tiles_gdf,
    config: Dict,
    raster_img: DatasetReader
) -> None:
    """
    Run model inference and write predictions to raster files.
    Supports resampling logits to output_px_meters if different from reference_resolution.
    """
    device = config['device']
    margin_px = config['margin']
    tile_size = config['img_pixels_detection']
    output_type = config['output_type']
    ref_res = config['reference_resolution']
    out_res = config.get('output_px_meters', ref_res)  # fallback to ref_res if not set
    needs_rescale = abs(ref_res - out_res) > 1e-6
    image_bounds = config['image_bounds']

    logger.info("\n[ ] Starting inference and writing raster tiles...\n")

    transform = raster_img.transform
    raster_result_shape = (model.task_nclasses,raster_img.shape[0],raster_img.shape[1])
    raster_logits = np.zeros(raster_result_shape, dtype=np.int8)

    for batch in tqdm(dataloader, file=sys.stdout):
        inputs = {
            mod: batch[mod].to(device)
            for mod in batch if mod not in ['index'] and not mod.endswith('_DATES')
        }
        for mod in batch:
            if mod.endswith('_DATES'):
                inputs[mod] = batch[mod].to(device)

        indices = batch['index'].cpu().numpy().flatten()
        rows = tiles_gdf.iloc[indices]

        with torch.no_grad():
            logits_tasks, _ = model(inputs)

        for task_name, logits in logits_tasks.items():
            logits = logits.cpu().numpy()

            for i, idx in enumerate(indices):
                
                row = rows.iloc[i]

                logit_patch = logits[i, :, margin_px:tile_size - margin_px, margin_px:tile_size - margin_px]
                res_ref = config["reference_resolution"]
                res_out = config.get("output_px_meters", res_ref)
                needs_rescale = abs(res_out - res_ref) > 1e-6
                scale = res_ref / res_out if needs_rescale else 1.0

                if needs_rescale:
                    logit_patch = resample_prediction(logit_patch, scale)
                    
                prediction = convert(logit_patch, output_type)  # (C, H, W) dtypes to int8

                # Top-left corner in output raster (CRS coordinates to pixels)
                left = row['left']
                top = row['top']
                left_px = int(round((left - image_bounds['left']) / out_res))
                top_px  = int(round((top - image_bounds['top']) / out_res))

                c, h, w = prediction.shape

                # Output raster dimensions
                img_height = raster_logits.shape[1]
                img_width  = raster_logits.shape[2]

                # Output raster dimensions
                img_height = int(round((image_bounds['top'] - image_bounds['bottom']) / out_res))
                img_width  = int(round((image_bounds['right'] - image_bounds['left']) / out_res))

                 # --- SAFETY CLIPPING ---
                x1 = max(0, left_px)
                y1 = max(0, top_px)
                x2 = min(img_width, left_px + w)
                y2 = min(img_height, top_px + h)

                # Check if anything remains inside the raster
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"[!] Tile {row['id']} fully outside raster bounds. Skipping.")
                    continue

                # Crop the prediction accordingly
                dx1 = x1 - left_px  # how much we cut from left of the patch
                dy1 = y1 - top_px   # how much we cut from top of the patch
                dx2 = dx1 + (x2 - x1)
                dy2 = dy1 + (y2 - y1)

                cropped_pred = prediction[:, dy1:dy2, dx1:dx2]
                
                raster_logits[:, y1:y2, x1:x2] += cropped_pred
    
    return raster_logits, transform

def logits_to_labels_and_confidence(probs):
    """
    
    """
    labels = np.argmax(probs, axis=0).astype(np.uint8)
    confidence = np.max(probs, axis=0)
    return labels, confidence

def vectorize_segmentation(labels, confidence, transform, crs="EPSG:5490", simplification_tolerance=1.0):
    """
    labels: HxW np.ndarray
    confidence: HxW np.ndarray
    transform: affine transform of full raster
    class_map: {int: str} mapping class IDs → names
    """
    polygons = []
    for geom, value in tqdm(shapes(labels, transform=transform)):
        value = int(value)
        if value == 0:
            continue  # skip background
        poly = shape(geom)
        poly = poly.simplify(simplification_tolerance, preserve_topology=True)
        mean_conf = float(confidence[labels == value].mean())
        polygons.append({
            "geometry": poly,
            "class_id": value,
            #"class_name": class_map.get(value, str(value)),
            "confidence": mean_conf
        })
    return gpd.GeoDataFrame(polygons, crs=crs)


def _vectorize_single_class(args):
    class_id, labels, confidence, transform, simplification_tolerance, min_area = args
    from shapely.geometry import shape
    from rasterio.features import shapes

    mask = labels == class_id
    polygons = []
    for geom, _ in tqdm(shapes(mask.astype(np.uint8), mask=mask, transform=transform),desc=f"vectorizing segmentation class : {class_id}"):
        poly = shape(geom)
        if not poly.is_valid or poly.is_empty or poly.area < min_area:
            continue
        poly = poly.simplify(simplification_tolerance, preserve_topology=True)
        mean_conf = float(confidence[mask].mean())
        polygons.append({"geometry": poly, "class_id": int(class_id), "confidence": mean_conf})
    return polygons

from multiprocessing import Pool

def vectorize_segmentation_parallel(labels, confidence, transform, n_jobs=4, **kwargs):
    class_ids = np.unique(labels)
    class_ids = class_ids[class_ids != 0]

    with Pool(processes=n_jobs) as pool:
        results = pool.map(_vectorize_single_class, [
            (cid, labels, confidence, transform, kwargs.get("simplification_tolerance", 1.0), kwargs.get("min_area", 4.0))
            for cid in class_ids
        ])

    polygons = [p for sublist in results for p in sublist]
    if len(polygons)>0:
        return gpd.GeoDataFrame(polygons, crs=kwargs.get("crs", "EPSG:5490"))
    else:
        return gpd.GeoDataFrame()


def postpro_outputs(temp_paths: Dict[str, str], config: Dict) -> None:
    """
    Convert output rasters to Cloud Optimized GeoTIFFs (COG) if requested.
    """
    if config.get("cog_conversion", False):
        for task_name, temp_path in temp_paths.items():
            cog_path = temp_path.replace(".tif", "_COG.tif")
            convert_to_cog(temp_path, cog_path)
            logger.info(f"\n[✓] Converted to COG: {cog_path}")


def run_inference(config_path: str) -> None:
    """
    Main entry point to run inference from a config file.
    """

    start_total = time.time()
    config = prep_config(config_path)
    start_slice = time.time()
    tiles_gdf = generate_patches_from_reference(config)
    logger.info(f"[✓] Sliced into {len(tiles_gdf)} tiles in {time.time() - start_slice:.2f}s")

    start_model = time.time()
    patch_sizes = compute_patch_sizes(config)

    model = build_inference_model(config, patch_sizes).to(config['device'])
    logger.info(f"[✓] Loaded model and checkpoint in {time.time() - start_model:.2f}s")

    dataset = prep_dataset(config, tiles_gdf, patch_sizes)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_worker'])

    ref_img = rasterio.open(config['modalities'][config['reference_modality']]['input_img_path'])
    output_files, temp_paths = init_outputs(config, ref_img)

    start_infer = time.time()
    inference_and_write(model, dataloader, tiles_gdf, config, output_files, ref_img)
    logger.info(f"[✓] Inference completed in {time.time() - start_infer:.2f}s")

    postpro_outputs(temp_paths, config)
    
    logger.info(f"\n[✓] Total time: {time.time() - start_total:.2f}s")
    logger.info(f"\n[✓] Inference complete. Rasters written to: {list(temp_paths.values())}\n")


from pathlib import Path
import os
import hashlib
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import f1_score, accuracy_score
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
from scipy import ndimage
from rasterio.features import rasterize

DEBUG_IMG_DIR = "/app/data/datasets/debug/bush/debug" 


def cache_key(row):
    h = hashlib.md5(row.image_path.encode()).hexdigest()[:8]
    return f"{row.name}_{h}.pt"

def build_zones(buildings_geom, limit_geometry, distances=(10, 30, 50)):
    """
    Build concentric buffer zones around buildings, clipped by a limit geometry.

    Parameters
    ----------
    buildings_geom : shapely.geometry
        Base geometry (e.g. buildings)
    limit_geometry : shapely.geometry
        Geometry used to clip the zones
    distances : tuple
        Buffer distances in meters (must start with 0)

    Returns
    -------
    dict[str, shapely.geometry]
        Zones z0, z1, ...
    """

    # Defensive cleanup
    buildings_geom = buildings_geom.buffer(0)
    limit_geometry = limit_geometry.buffer(0)

    zones = {}

    # Precompute buffers
    buffers = {d: buildings_geom.buffer(d) for d in distances}

    for i, d in enumerate(distances):
        zone_name = f"z{d}"

        if i == 0:
            inner = buildings_geom
        else:
            inner = buffers[distances[i - 1]]
        outer = buffers[d]
        zone = outer.difference(inner)

        # Clip to limit geometry
        zone = zone.intersection(limit_geometry)

        # Clean invalid / empty geometries
        if zone.is_empty:
            zones[zone_name] = None
        else:
            zones[zone_name] = zone.buffer(0)

    return zones

def _fallback_square(geometry, area_m2=1):
    """
    Create a square of given area (m²) centered on geometry barycenter.
    """
    center = geometry.centroid
    side = np.sqrt(area_m2)
    half = side / 2.0

    return box(
        center.x - half,
        center.y - half,
        center.x + half,
        center.y + half
    )

def max_continuous_surface(data, threshold, pixel_area):
    """
    data : (H, W)
    """
    binary = data > threshold

    labeled, n = ndimage.label(binary)
    if n == 0:
        return 0.0

    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    return np.max(sizes) * pixel_area

def nb_coherent_surface(data, threshold, pixel_area, area_m2_threshold=5):
    """
    data : (H, W)
    """
    binary = data > threshold

    structure = np.ones((3, 3))  # 8-connectivity
    labeled, num_groups = ndimage.label(binary, structure=structure)

    pixel_counts = ndimage.sum(
        np.ones_like(labeled),
        labeled,
        index=range(1, num_groups + 1)
    )

    surfaces_m2 = pixel_counts * pixel_area
    
    nb_representatives_surfaces = len([x for x in surfaces_m2 if x>area_m2_threshold])

    return nb_representatives_surfaces

def compute_stats(data, pixel_area):
    """
    data : 2D np.array (H, W) with NaNs
    pixel_area : surface of one pixel (m²)
    """

    valid = data[~np.isnan(data)]
    total_pixels = valid.size

    if total_pixels == 0:
        return {
            "mean": 0,
            "nb_surf_sup1m2_100": 0,
            "nb_surf_sup1m2_200": 0,
            "surf_gt_100_ratio": 0.0,
            "surf_gt_200_ratio": 0.0,
            #"count_gt_50_ratio": 0.0,
            "count_gt_100_ratio": 0.0,
            #"count_gt_150_ratio": 0.0,
            "count_gt_200_ratio": 0.0,
        }

    # Quantiles
    q10 = np.percentile(valid, 10)
    q90 = np.percentile(valid, 90)

    low_decile_mean = valid[valid <= q10].mean()
    high_decile_mean = valid[valid >= q90].mean()

    # Total surface
    total_surface = total_pixels * pixel_area

    # Continuous surfaces
    surf_gt_100 = max_continuous_surface(data, 100, pixel_area)
    surf_gt_200 = max_continuous_surface(data, 200, pixel_area)
    nb_sup_5m2_surfaces_100 = nb_coherent_surface(data, 100, pixel_area, area_m2_threshold=5)
    nb_sup_5m2_surfaces_200 = nb_coherent_surface(data, 200, pixel_area, area_m2_threshold=5)
    
    return {
        "mean": valid.mean()/255,
        #"low_decile_mean": low_decile_mean/255,
        #"high_decile_mean": high_decile_mean/255,

        # normalized continuous surfaces
        "surf_gt_100_ratio": surf_gt_100 / total_surface,
        "surf_gt_200_ratio": surf_gt_200 / total_surface,
        "nb_surf_sup1m2_100": nb_sup_5m2_surfaces_100, 
        "nb_surf_sup1m2_200": nb_sup_5m2_surfaces_200, 
        # normalized pixel counts
        #"count_gt_50_ratio": np.sum(valid > 50) / total_pixels,
        "count_gt_100_ratio": np.sum(valid > 100) / total_pixels,
        #"count_gt_150_ratio": np.sum(valid > 150) / total_pixels,
        "count_gt_200_ratio": np.sum(valid > 200) / total_pixels
    }
    
def extract_building_zone(
    src_path,
    geometry,
    band_index=0,
    threshold=150,
    min_area_m2=20.0,
    fallback_square_m2=1.0
):
    """
    Extract merged continuous zones above threshold from a raster band.

    Returns
    -------
    shapely.geometry
    """

    with rasterio.open(src_path) as src:
        pixel_area = abs(src.transform.a * src.transform.e)

        try:
            data, transform = mask(
                src,
                [geometry],
                crop=True,
                filled=True,
                nodata=0
            )
        except Exception:
            return _fallback_square(geometry, fallback_square_m2)

        band = data[band_index]

        # -----------------------------
        # Threshold
        # -----------------------------
        binary = band >= threshold

        if not binary.any():
            return _fallback_square(geometry, fallback_square_m2)

        # -----------------------------
        # Connected components (8-connectivity)
        # -----------------------------
        structure = np.ones((3, 3), dtype=np.int8)
        labeled, n_labels = ndimage.label(binary, structure=structure)

        if n_labels == 0:
            return _fallback_square(geometry, fallback_square_m2)

        # -----------------------------
        # Extract valid components
        # -----------------------------
        valid_geoms = []

        for label_id in range(1, n_labels + 1):
            mask_cc = labeled == label_id
            area_m2 = mask_cc.sum() * pixel_area

            if area_m2 < min_area_m2:
                continue

            for geom, val in shapes(
                mask_cc.astype(np.uint8),
                mask=mask_cc,
                transform=transform
            ):
                if val == 1:
                    valid_geoms.append(shape(geom))

        if not valid_geoms:
            return _fallback_square(geometry, fallback_square_m2)

        # -----------------------------
        # Merge all valid components
        # -----------------------------
        return unary_union(valid_geoms)


def extract_zone_raster(src, geom, bands_idx):
    if geom.is_empty:
        return None

    out, _ = mask(
        src,
        [mapping(geom)],
        crop=True,
        nodata=src.nodata
    )

    # sélection bandes 12,13,14
    data = out[bands_idx, :, :].astype(float)

    if src.nodata is not None:
        data[data == src.nodata] = np.nan

    return data


def debug_geometries_with_a_tiffprint(src_path, building_geom, geometry, zones, sample_id, pixel_size=0.2*0.2 ):
        debug_dir = Path(DEBUG_IMG_DIR)
        debug_dir.mkdir(parents=True, exist_ok=True)

        debug_path = debug_dir / f"{sample_id}_zones_debug.tif"

        with rasterio.open(src_path) as src:
            out_image, out_transform = mask(
                src,
                [geometry],
                crop=True,
                filled=True,
                nodata=255
            )

            height, width = out_image.shape[1:]

            # -----------------------------
            # Grayscale encoding
            # -----------------------------
            zone_values = {
                "z0": 0,     # building
                "z1": 60,
                "z2": 110,
                "z3": 170,
                "z4": 230,
            }
        
            # -----------------------------
            # Prepare shapes to rasterize
            # -----------------------------
            shapes_to_burn = []

            # Building
            if building_geom and not building_geom.is_empty:
                shapes_to_burn.append((building_geom, zone_values["z0"]))

            # buffer zones used for features
            for zone_name, zone_geom in zones.items():
                if zone_geom and not zone_geom.is_empty:
                    zone_color_value = zone_values.get(zone_name, 255)
                    shapes_to_burn.append((zone_geom, zone_color_value))

            # -----------------------------
            # Rasterize
            # -----------------------------
            debug_raster = rasterize(
                shapes=shapes_to_burn,
                out_shape=(height, width),
                transform=out_transform,
                fill=255,
                dtype="uint8"
            )

            # -----------------------------
            # Write GeoTIFF
            # -----------------------------
            profile = src.profile.copy()
            profile.update({
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "uint8",
                "transform": out_transform,
                "nodata": 255,
            })

            with rasterio.open(debug_path, "w", **profile) as dst:
                dst.write(debug_raster, 1)

def extract_features(src_path, geometry, gdf_forest_zones=None, gdf_water_zones=None, gdf_u_zones=None, sample_id=None,  debug=False):
    """

    """

    building_geom = extract_building_zone(src_path, geometry)

    zones = build_zones(building_geom, geometry)
    
    # give me a print image here in a tiff file to control intermediate results
    if debug:
        debug_geometries_with_a_tiffprint(src_path, building_geom, geometry, zones, sample_id)
    
    features_bands = [12,13,14]
    results = []
    with rasterio.open(src_path) as src:
        pixel_area = abs(src.transform.a * src.transform.e)
        for zone_name, zone_geom in zones.items():
            print(f"Starting data extraction on {zone_geom} --- from {src} - crs {src.crs} ")
            try :
                
                data = extract_zone_raster(src, zone_geom, features_bands)
                print(f"Data extracted for row {sample_id} zone {zone_name} ")
            except:
                print(f"No intersection between bat and raster segmentation source")
                data = None

            for i, band in enumerate(features_bands):
                if data is None:
                    stats = {
                        "mean": 0,
                        "surf_gt_100_ratio": 0.0,
                        "surf_gt_200_ratio": 0.0,
                        "nb_surf_sup1m2_100": 0,
                        "nb_surf_sup1m2_200": 0,
                        #"count_gt_50_ratio": 0.0,
                        "count_gt_100_ratio": 0.0,
                        #"count_gt_150_ratio": 0.0,
                        "count_gt_200_ratio": 0.0,
                        }
                else:
                    stats = compute_stats(data[i], pixel_area)
                    print(f"Features built for : row {sample_id} - zone {zone_name} - band {band}")
                results.append({
                    "sample_id": sample_id,
                    "zone": zone_name,
                    "band": band,
                    **stats
                    })
    df_stats = pd.DataFrame(results)
    df_stats = df_stats.drop_duplicates(subset=['sample_id','zone','band'], keep='first')

    metrics = [
    "mean",
    "surf_gt_100_ratio",
    "surf_gt_200_ratio",
    "nb_surf_sup1m2_100",
    "nb_surf_sup1m2_200",
    #"count_gt_50_ratio",
    "count_gt_100_ratio",
    #"count_gt_150_ratio",
    "count_gt_200_ratio"
    ]

    df_wide = (
        df_stats
        .set_index(["sample_id", "zone", "band"])[metrics]
        .unstack(["zone", "band"])
    )

    # Flatten multi-index columns
    df_wide.columns = [
        f"{zone}_{band}_{metric}"
        for metric, zone, band in df_wide.columns
    ]

    df_wide = df_wide.reset_index()

    df_wide['surf_area'] = df_wide.geometry.area


    if building_geom.area > 10 :
        df_wide['has_building'] = [1]
    else:
        df_wide['has_building'] = [0]
       
    if building_geom.area > 40:
        df_wide['has_inhabited_building'] = [1]
    else:
        df_wide['has_inhabited_building'] = [0] 
    
    
    buffered_for_forest = geometry.buffer(5)

    # Spatial feature on intersections with dangerous forest zones
    candidates = gdf_forest_zones[gdf_forest_zones.intersects(buffered_for_forest)]
    
    if len(candidates)==0:
        df_wide['has_contact_forest_zone'] = [0]
    else: 
        df_wide['has_contact_forest_zone'] = [1]
        
    # Spatial feature on intersections with rivers zones 
    buffered_for_rivers = geometry.buffer(20)
    
    candidates = gdf_water_zones[gdf_water_zones.intersects(buffered_for_rivers)]
    
    if len(candidates)==0:
        df_wide['has_contact_river_zone'] = [0]
    else: 
        df_wide['has_contact_river_zone'] = [1]

    # Spatial feature on intersections with u zones 
    buffered_for_u = geometry.buffer(0)
    
    candidates = gdf_u_zones[gdf_u_zones.intersects(buffered_for_u)]
    
    if len(candidates)==0:
        df_wide['has_contact_u_zone'] = [0]
    else: 
        df_wide['has_contact_u_zone'] = [1]


    return df_wide

def postprocess_pred_control(test_gdf: gpd.GeoDataFrame, x_test_business_features: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Post-process pred_control according to spatial constraints.
    """

    gdf = pd.concat([test_gdf.reset_index(drop=True),x_test_business_features.reset_index(drop=True)],axis=1)

    # Initialize post-processed prediction
    gdf["pred_control_pp"] = gdf["pred_control"]

    # Rule 1: remove if contact with river
    mask_river = (
        (gdf["pred_control_pp"] == 1) &
        (gdf["has_contact_river_zone"] == 1)
    )
    gdf.loc[mask_river, "pred_control_pp"] = 0

    # Rule 2: remove if NO contact with forest
    mask_forest = (
        (gdf["pred_control_pp"] == 1) &
        (gdf["has_contact_forest_zone"] == 0)
    )
    gdf.loc[mask_forest, "pred_control_pp"] = 0

    # Rule 3: remove if is smaller than 200m²
    mask_small_zone = (
        (gdf["pred_control_pp"] == 1) &
        (gdf.geometry.area < 200)
    )
    gdf.loc[mask_small_zone, "pred_control_pp"] = 0

    # Rule 4 : set all u_zone_old to 0 except if it has contact with forest and has a neighbourg positive to control
    # TODO : missing info on "set all u_zone_old to 0 except"
    
    #build flag rule
    gdf_buffer = gdf.copy()
    gdf_buffer["geometry"] = gdf.geometry.buffer(2)

    joined = gpd.sjoin(
        gdf_buffer,
        gdf[["geometry", "pred_control_pp"]],
        how="left",
        predicate="intersects"
    )

    joined = joined[joined.index != joined.index_right]

    neighbor_flag = (
        joined.groupby(joined.index)["pred_control_pp"]
        .apply(lambda x: (x == 1).any())
    )

    gdf["has_rule_u_zone"] = (
        gdf["has_contact_forest_zone"] &
        gdf.index.map(neighbor_flag).fillna(False)
    ).astype(int)
    
    # apply rule
    mask_u_zone= (gdf["has_rule_u_zone"] == 1)
    gdf.loc[mask_u_zone, "pred_control_pp"] = 1

    
    return gdf

                 
def find_best_threshold(y_true, y_proba, metric="f1", n_steps=100):
    """
    Find best threshold for binary predictions based on a metric.

    Parameters
    ----------
    y_true : np.ndarray or list
        True binary labels (0/1)
    y_proba : np.ndarray or list
        Predicted probabilities for positive class
    metric : str
        "f1" or "accuracy"
    n_steps : int
        Number of thresholds to test between 0 and 1

    Returns
    -------
    best_threshold : float
    best_score : float
    """
    thresholds = np.linspace(0, 1, n_steps)
    best_score = -np.inf
    best_threshold = 0.5  # default fallback

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    for t in thresholds:
        y_pred = (y_proba > t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score

def get_or_build_xy(set_gdf, set_name, gdf_forests, gdf_waters, gdf_u_zones, cache_dir: str = None, debug=True):
    df_features_list = []
    if debug:
        set_gdf = set_gdf[:min(10,len(set_gdf))]
    
    cache_filename = os.path.join(cache_dir, set_name + '.parquet')
    if not os.path.exists(cache_filename):
        for row in set_gdf.iterrows():
            df_feature_row = extract_features(row[1].image_path, row[1].geometry, gdf_forest_zones=gdf_forests, gdf_water_zones=gdf_waters , gdf_u_zones=gdf_u_zones, sample_id=row[0],debug=debug)
            df_feature_row['target_control'] = row[1].target_control
            df_features_list.append(df_feature_row)
            
        df_set = pd.concat(df_features_list)
        df_set.to_parquet(cache_filename)
    else :
        df_set = pd.read_parquet(cache_filename)
    y = df_set['target_control']
    x = df_set.drop(columns=['target_control','sample_id'])
    
    return x, y

def preprocess_features(set_gdf, gdf_forests, gdf_waters, gdf_u_zones, cache_dir: str = None, debug=True):
    df_features_list = []

    if debug:
        set_gdf = set_gdf[:min(10,len(set_gdf))]
    
    cache_filename = os.path.join(cache_dir,'set_features.parquet')
    if not os.path.exists(cache_filename):    

        for row in set_gdf.iterrows():
            
            df_feature_row = extract_features(row[1].image_path, row[1].geom, gdf_forest_zones=gdf_forests, gdf_water_zones=gdf_waters, gdf_u_zones=gdf_u_zones, sample_id=row[0],debug=debug)
            df_features_list.append(df_feature_row)
            
        df_set = pd.concat(df_features_list)
        df_set.to_parquet(cache_filename)
    else :
        df_set = pd.read_parquet(cache_filename)
    
    x_train_business_features = df_set[['has_contact_river_zone','has_contact_forest_zone','has_contact_u_zone','has_inhabited_building','has_building']]
    x_train_ml_features = df_set.drop(columns=['has_contact_river_zone','has_contact_forest_zone','sample_id'])
    
    return x_train_ml_features, x_train_business_features
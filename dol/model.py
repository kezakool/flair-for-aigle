import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import rasterio
from xgboost import XGBClassifier
from shapely import geometry

from utils.dol import preprocess_features, postprocess_pred_control
from utils.logs import update_progress


logger = logging.getLogger(__name__)


class DOLClassifier:
    """
    DOL classification pipeline.

    Responsibilities:
        - Initialize the XGBoost DOL classifier.
        - Load and prepare external geospatial sources.
        - Preprocess ML and business features.
        - Run model inference.
        - Apply business rules.
        - Report progression during long-running operations.
    """

    def __init__(
        self,
        model_path: str,
        model_metadata_path: str,
        geozone_code: str,
        work_folder: str,
        debug: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> None:
        self.model_path: str = model_path
        self.model_metadata_path: str = model_metadata_path
        self.geozone_code: str = geozone_code
        self.work_folder: str = work_folder
        self.debug: bool = debug

        self.model: Optional[XGBClassifier] = None
        self.threshold: Optional[float] = None

        # Optional callback allowing the caller to plug its own
        # progression mechanism.
        self.progress_callback = progress_callback

        self._initialize_model()

    # ------------------------------------------------------------------
    # Progression
    # ------------------------------------------------------------------

    def _update_progress(
        self,
        progress: int,
        status: str,
    ) -> None:
        """
        Update the DOL classification progression.

        Args:
            progress: Progress percentage between 0 and 100.
            status: Human-readable current processing status.
        """

        progress = max(0, min(100, progress))

        logger.info(
            "DOL classification progression: %s%% - %s",
            progress,
            status,
        )

        if self.progress_callback is not None:
            self.progress_callback(progress, status)
        else:
            update_progress(progress, status)

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _initialize_model(self) -> None:
        """Load the XGBoost model and its metadata."""

        self._update_progress(
            0,
            "initializing DOL classifier",
        )

        logger.info("Loading DOL XGBoost model...")

        self.model = XGBClassifier()
        self.model.load_model(self.model_path)

        self._update_progress(
            10,
            "DOL model loaded",
        )

        logger.info("Loading DOL model metadata...")

        with open(self.model_metadata_path, "r") as metadata_file:
            metadata: Dict[str, Any] = json.load(metadata_file)

        self.model.scale_pos_weight = metadata["scale_pos_weight"]
        self.threshold = float(metadata["decision_threshold"])

        self._update_progress(
            15,
            "DOL classifier initialized",
        )

        logger.info(
            "DOL classifier initialized with decision threshold %.4f",
            self.threshold,
        )

    # ------------------------------------------------------------------
    # External sources
    # ------------------------------------------------------------------

    def _load_external_sources(
        self,
        dol_specific_local_sources: Dict[str, str],
    ) -> Tuple[
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
    ]:
        """
        Load external geospatial datasets required by DOL processing.

        Returns:
            DOL zones, forest zones, water zones and urban zones.
        """

        logger.info("Loading DOL external sources...")

        gdf_dol_ilots: gpd.GeoDataFrame = gpd.read_file(
            dol_specific_local_sources["db_zone_dol"]
        )

        gdf_dol_ilots_geozone: gpd.GeoDataFrame = (
            gdf_dol_ilots[
                gdf_dol_ilots.insee_com == self.geozone_code
            ]
        )

        self._update_progress(
            20,
            "DOL zones loaded",
        )

        gdf_forests_zones: gpd.GeoDataFrame = gpd.read_file(
            dol_specific_local_sources["db_forest_path"]
        )

        gdf_waters_zones: gpd.GeoDataFrame = gpd.read_file(
            dol_specific_local_sources["db_waters_path"]
        )

        gdf_u_zone: gpd.GeoDataFrame = gpd.read_file(
            dol_specific_local_sources["db_zone_urba_path"]
        )

        self._update_progress(
            25,
            "DOL external sources loaded",
        )

        return (
            gdf_dol_ilots_geozone,
            gdf_forests_zones,
            gdf_waters_zones,
            gdf_u_zone,
        )

    # ------------------------------------------------------------------
    # Image bounds
    # ------------------------------------------------------------------

    def _build_image_bounds(
        self,
        result_folder: str,
    ) -> gpd.GeoDataFrame:
        """
        Build a GeoDataFrame containing the spatial extent
        of segmentation result rasters.
        """

        logger.info("Building segmentation image bounds...")

        result_segmentation_files = [
            file_name
            for file_name in os.listdir(result_folder)
            if file_name.endswith(".tif")
        ]

        imgs_bounds = []

        total_images: int = len(result_segmentation_files)

        for index, img_path in enumerate(result_segmentation_files):
            full_path: str = os.path.join(
                result_folder,
                img_path,
            )

            with rasterio.open(full_path) as src:
                bbox = src.bounds
                bbox_polygon = geometry.box(*bbox)

            imgs_bounds.append(
                [full_path, bbox_polygon]
            )

            # Image-bound processing occupies 25 -> 30%.
            if total_images > 0:
                progress: int = 25 + int(
                    ((index + 1) / total_images) * 5
                )
                self._update_progress(
                    progress,
                    f"processing segmentation bounds "
                    f"({index + 1}/{total_images})",
                )

        gdf_img: gpd.GeoDataFrame = gpd.GeoDataFrame(
            data=imgs_bounds,
            columns=["image_path", "geometry"],
            geometry="geometry",
            crs="EPSG:2154",
        )

        gdf_img.to_crs(
            "EPSG:2154",
            inplace=True,
        )

        gdf_img.drop_duplicates(
            subset="image_path",
            inplace=True,
        )

        return gdf_img

    # ------------------------------------------------------------------
    # Geozone feature preparation
    # ------------------------------------------------------------------

    def _build_geozone_features(
        self,
        result_folder: str,
        gdf_dol_ilots_geozone: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Build the geozone dataset used as input for feature preprocessing.
        """

        logger.info("Building DOL geozone features...")

        gdf_img: gpd.GeoDataFrame = self._build_image_bounds(
            result_folder
        )

        gdf_geozone_data: gpd.GeoDataFrame = gpd.sjoin(
            gdf_img,
            gdf_dol_ilots_geozone,
            how="right",
            predicate="intersects",
        ).drop(
            columns="index_left"
        )

        gdf_geozone_data = gdf_geozone_data[
            ~gdf_geozone_data.image_path.isna()
        ]

        gdf_geozone_data.rename(
            columns={"geom": "geometry"},
            inplace=True,
        )

        self._update_progress(
            35,
            "DOL geozone features prepared",
        )

        return gdf_geozone_data

    # ------------------------------------------------------------------
    # Feature preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        result_folder: str,
        dol_specific_local_sources: Dict[str, str],
    ) -> Tuple[
        gpd.GeoDataFrame,
        Any,
        Any,
    ]:
        """
        Prepare all ML and business features.

        Returns:
            gdf_geozone_data:
                Geospatial DOL entities used during classification.

            x_ml_features:
                Features passed to the XGBoost model.

            x_business_features:
                Features passed to the business postprocessing.
        """

        logger.info(
            "Starting DOL classifier feature preprocessing..."
        )

        self._update_progress(
            16,
            "loading DOL external sources",
        )

        (
            gdf_dol_ilots_geozone,
            gdf_forests_zones,
            gdf_waters_zones,
            gdf_u_zone,
        ) = self._load_external_sources(
            dol_specific_local_sources
        )

        self._update_progress(
            27,
            "building DOL geozone features",
        )

        gdf_geozone_data: gpd.GeoDataFrame = (
            self._build_geozone_features(
                result_folder,
                gdf_dol_ilots_geozone,
            )
        )

        logger.info(
            "Running DOL feature preprocessing..."
        )

        self._update_progress(
            40,
            "computing DOL classifier features",
        )

        cache_dir: str = os.path.join(
            self.work_folder,
            "cache",
        )

        x_ml_features, x_business_features = preprocess_features(
            gdf_geozone_data,
            gdf_forests_zones,
            gdf_waters_zones,
            gdf_u_zone,
            cache_dir=cache_dir,
            debug=self.debug,
        )

        self._update_progress(
            55,
            "DOL classifier features ready",
        )

        logger.info(
            "DOL feature preprocessing completed."
        )

        return (
            gdf_geozone_data,
            x_ml_features,
            x_business_features,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        gdf_geozone_data: gpd.GeoDataFrame,
        x_ml_features: Any,
    ) -> gpd.GeoDataFrame:
        """
        Run XGBoost inference and apply the configured threshold.

        Args:
            gdf_geozone_data: DOL entities being classified.
            x_ml_features: Preprocessed ML features.

        Returns:
            GeoDataFrame containing probability and binary prediction.
        """

        if self.model is None:
            raise RuntimeError(
                "DOL classifier model has not been initialized."
            )

        if self.threshold is None:
            raise RuntimeError(
                "DOL classifier decision threshold has not been initialized."
            )

        logger.info(
            "Starting DOL classifier inference..."
        )

        self._update_progress(
            60,
            "running DOL classifier inference",
        )

        y_proba = self.model.predict_proba(
            x_ml_features
        )[:, 1]

        self._update_progress(
            75,
            "DOL classifier inference completed",
        )

        gdf_geozone_data["proba_control"] = y_proba

        gdf_geozone_data["pred_control"] = (
            gdf_geozone_data["proba_control"]
            >= self.threshold
        ).astype(int)

        gdf_geozone_data.set_geometry(
            "geometry",
            inplace=True,
        )

        logger.info(
            "DOL classifier predictions generated."
        )

        return gdf_geozone_data

    # ------------------------------------------------------------------
    # Business processing
    # ------------------------------------------------------------------

    def apply_business_rules(
        self,
        gdf_geozone_data: gpd.GeoDataFrame,
        x_business_features: Any,
    ) -> gpd.GeoDataFrame:
        """
        Apply existing DOL business rules to classifier predictions.

        Args:
            gdf_geozone_data: GeoDataFrame containing ML predictions.
            x_business_features: Features required by business rules.

        Returns:
            Final DOL classification GeoDataFrame.
        """

        logger.info(
            "Starting DOL business processing..."
        )

        self._update_progress(
            80,
            "applying DOL business rules",
        )

        result: gpd.GeoDataFrame = postprocess_pred_control(
            gdf_geozone_data,
            x_business_features,
        )

        self._update_progress(
            95,
            "DOL business processing completed",
        )

        return result

    # ------------------------------------------------------------------
    # Complete pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        result_folder: str,
        dol_specific_local_sources: Dict[str, str],
    ) -> gpd.GeoDataFrame:
        """
        Execute the complete DOL classification pipeline.

        Args:
            result_folder:
                Folder containing segmentation results.

            dol_specific_local_sources:
                Paths to the external DOL/forest/water/urban datasets.

        Returns:
            Final DOL classification GeoDataFrame.
        """

        logger.info(
            "Starting complete DOL classification pipeline..."
        )

        self._update_progress(
            15,
            "starting DOL classification",
        )

        (
            gdf_geozone_data,
            x_ml_features,
            x_business_features,
        ) = self.preprocess(
            result_folder=result_folder,
            dol_specific_local_sources=dol_specific_local_sources,
        )

        gdf_geozone_data = self.predict(
            gdf_geozone_data=gdf_geozone_data,
            x_ml_features=x_ml_features,
        )

        result: gpd.GeoDataFrame = self.apply_business_rules(
            gdf_geozone_data=gdf_geozone_data,
            x_business_features=x_business_features,
        )

        self._update_progress(
            100,
            "DOL classification completed",
        )

        logger.info(
            "DOL classification pipeline completed."
        )

        return result
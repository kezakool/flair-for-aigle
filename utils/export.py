from shapely import box
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from utils.utils import *
from shapely.wkt import loads
from sqlalchemy import *
from geoalchemy2 import Geometry
import datetime
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class Exporter(object):
    """
    The `Exporter` class initializes various resources required for exporting data, such as loading
    official buildings, category zones, relevant detections, parcels, and tiles.
    """

    def __init__(self, detection_crs) -> None:
        """
        Initializes various resources required for the Exporter, processing detection data, including
        loading official buildings, category zones, parcels, and tiles.
        
        :param detection_crs: The `detection_crs` parameter in the `__init__` method is used to specify
        the coordinate reference system (CRS) that will be used for the detection data. It is important
        for ensuring that all spatial data is correctly aligned and projected in the same CRS for
        accurate analysis and visualization
        """
        self.export_batch_size = 50000
        load_dotenv()
        self.db_string_sia = os.getenv("DB_STRING_SIA_PROD")
        self.db_string_aigle = os.getenv("DB_STRING_PROD")
        self.detection_crs = detection_crs
        
    def transform_to_inference_table(self, gdf_detections, batch_id):
        
        gdf_detections = gdf_detections[['confidence','class_id','geometry']]
        gdf_detections['batch_id'] = batch_id
        # Convert Polygon to WKT and include the SRID 
        gdf_detections['geometry'] = gdf_detections['geometry'].apply(lambda g: loads(g) if isinstance(g, str) else g)
        gdf_detections['geometry'] = gdf_detections['geometry'].apply(lambda geom: f"SRID={self.target_export_crs.replace('EPSG:','')};{geom.wkt}")
        
        gdf_detections.rename(columns={'confidence':'score','class_id':'object_type'}, inplace=True)
        return gdf_detections
    
    def insert_batch_object(self, conn, export_context):
        

        df_batchs = pd.read_sql('select * from detections.batch', con=conn)
        max_id = df_batchs['id'].max()
        self.batch_id = max_id + 1 if pd.notna(max_id) else 1
        sql = f"""INSERT INTO detections.batch (id, batch_name, created_at, model_id, batch_tiles_url, description) VALUES(nextval('detections.batch_id_seq'::regclass), '{export_context['batch_name']}', '{str(datetime.datetime.now())}', {export_context['model_id']}, '', '{export_context['description']}') RETURNING id;"""
        result = conn.execute(text(sql))
        batch_id = result.scalar()
        logger.info(f"batch initialized in db - id : {self.batch_id}, name : {export_context['batch_name']}")   
        return batch_id
    
    def insert_batch_detections(self, conn, gdf_detections, export_context, chunk_size=50000):

        self.gdf_detections = self.transform_to_inference_table(gdf_detections, batch_id=self.batch_id)
        nb_batchs = int(np.ceil(len(self.gdf_detections)/chunk_size))
        for i in range(0, len(self.gdf_detections), chunk_size):
            batch_num=int(i/chunk_size)+1
            logger.info(f"inserting inference batch {batch_num}/{nb_batchs} in db...")
            chunk = self.gdf_detections.iloc[i:i + chunk_size]
            chunk.to_sql('inference', con=conn, schema='detections',
                if_exists='append', index=False,dtype={'geometry': Geometry(geometry_type='POLYGON', srid=self.target_export_crs.replace('EPSG:',''))})

        logger.info("batch detections inserted in db")

    
    def export_to_aigle(self, gdf_detections, target_export_crs, output_folder, mapper, export_context):
        """_summary_

        Args:
            df_detections (_type_): _description_
            target_export_crs (_type_): _description_
            trusted_sources (_type_): _description_
        """
        self.target_export_crs = target_export_crs
        self.output_folder = output_folder
        self.mapper = mapper  
        
        # Create SQL connection engine
        self.engine = create_engine(self.db_string_aigle)

        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        self.gdf_detections = gdf_detections.copy()
        self.gdf_detections.to_crs(self.target_export_crs,inplace=True)
        # simplify and remap classes to aigle app classes
        self.gdf_detections['class_id'] = self.gdf_detections.class_id.apply(mapper.simplify_flair_classes_app)
        #remove unmapped classes 
        self.gdf_detections =  self.gdf_detections[self.gdf_detections.class_id != -1]
        
        # use app label as class id
        self.gdf_detections['class_id'] = self.gdf_detections.class_id.apply(mapper.map_aigle_classes_labels)
        
        # normalize scores by classes
        self.gdf_detections['confidence'] = self.gdf_detections.groupby('class_id')['confidence'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

        if export_context['export_sql']:
            # Establish a connection and use a transaction block
            try:
                with self.engine.connect() as conn:
                    with conn.begin() as transaction:  # Start the transaction
                        self.batch_id = self.insert_batch_object(conn, export_context)
                        self.insert_batch_detections(conn, self.gdf_detections, export_context)
            except SQLAlchemyError as e:
                # Log the error and ensure the transaction is rolled back
                logger.error(f"Transaction failed: {e}")
                raise  # Re-raise the exception to notify the caller of the failure
        else:
            self.gdf_detections = self.transform_to_inference_table(self.gdf_detections,batch_id=-1)
            
        self.gdf_detections['geometry'] = self.gdf_detections['geometry'].apply(lambda x : loads(x.split(';')[1]))
        self.gdf_detections = self.gdf_detections.set_geometry("geometry")
        self.gdf_detections = self.gdf_detections.set_crs(self.target_export_crs)
        gpkg_file_name = "batch_detections_" + export_context['batch_name'] + '.gpkg'
        exp_file = os.path.join(output_folder,gpkg_file_name)
        self.gdf_detections.to_file(exp_file, driver="GPKG")
        logger.info(f"batch detections saved to : {exp_file}")   


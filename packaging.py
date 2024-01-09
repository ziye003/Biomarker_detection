# Databricks notebook source
token = dbutils.secrets.get(scope="zi.ye", key="github_token")
%pip install git+https://dangeloaguirre:$token@github.com/sapientbio/peak-analysis

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Setup

# COMMAND ----------

import logging

import pandas as pd

from peak_analysis.extraction.packaging.configuration import DataPackagingConfig, DataPackagingInputs
from peak_analysis.extraction.packaging.run_packaging import DataPackager

import pyspark.sql.functions as sparkfn
import pyspark

# COMMAND ----------

logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s.%(msecs)03d:%(levelname)s:%(message)s',  # noqa: WPS323 E501
    datefmt='%Y-%m-%d %H:%M:%S',  # noqa: WPS323
)
_logger = logging.getLogger('peak_analysis')
_logger.setLevel(logging.DEBUG)

# COMMAND ----------

# MAGIC %md #Samples

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Filepaths

# COMMAND ----------

srjt = pd.read_excel('s3://hypoxia-pilot-study/01_raw_data/153sap23p001-hypoxia-pilot-metabolomics_sample_run_join_table.xlsx')
srjt.dropna(inplace=True)

# COMMAND ----------

display(srjt)

# COMMAND ----------

project_id = '153sap23p001'
project_bucket = 's3://hypoxia-pilot-study'
sample_type = 'feline_serum'
matrix_type = ''
neg_time_stamp = '2023-11-28_22-06-30.413610_UTC'
pos_time_stamp = '2023-11-28_22-18-34.105581_UTC'

# COMMAND ----------

peak_heights_neg_path = '/'.join([
    project_bucket,
    f'03_data_extraction/Feline_Serum_rLC_Neg_v3/neg_samples',
    f'{neg_time_stamp}/peak_heights.delta/',
])

peak_heights_pos_path = '/'.join([
    project_bucket,
    f'03_data_extraction/Feline_Serum_rLC_Pos_v3/pos_samples',
    f'{pos_time_stamp}/peak_heights.delta/',
])


peak_metadata_neg_path = peak_heights_neg_path.replace('peak_heights.delta/', 'peak_metadata.delta/')
peak_metadata_pos_path = peak_heights_pos_path.replace('peak_heights.delta/', 'peak_metadata.delta/')

output_path_internal = f'{project_bucket}/04_data_analysis/rLC/'
output_path_external = f'{project_bucket}/05_project_deliverables/rLC/'
name_prefix = f'{sample_type}_'

plate_conversion_table_path = 's3://database-prod-datalakehouse/utilities/96_to_384_well_conversion.csv'


# COMMAND ----------

inputs = DataPackagingInputs(
    peak_metadata_neg=spark.read.load(peak_metadata_neg_path),
    peak_metadata_pos=spark.read.load(peak_metadata_pos_path),
    peak_heights_neg=spark.read.load(peak_heights_neg_path),
    peak_heights_pos=spark.read.load(peak_heights_pos_path),
    sample_run_join_table=srjt,
    plate_conversion_table=pd.read_csv(plate_conversion_table_path),
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Packaging

# COMMAND ----------

_logger.info(f'Packaging data using internal config; {output_path_internal = }')

DataPackager(
    spark=spark,
    inputs=inputs,
    config=DataPackagingConfig.internal(
        name_prefix=name_prefix,
    ),
    output_path=output_path_internal,
).output_packaged()

# COMMAND ----------

_logger.info(f'Packaging data using external config; {output_path_external = }')

DataPackager(
    spark=spark,
    inputs=inputs,
    config=DataPackagingConfig.external(
        name_prefix=name_prefix,
    ),
    output_path=output_path_external,
).output_packaged()

# COMMAND ----------

feature_metadata = pd.read_csv(output_path_external + 'feature_metadata.csv')
feature_intensities = pd.read_csv(output_path_external + 'feature_intensities.csv')
sample_metadata = pd.read_csv(output_path_external + 'sample_metadata.csv')
feature_metadata.to_csv(output_path_external + f'{project_id}_{sample_type}_feature_metadata.csv', index=False)
feature_intensities.to_csv(output_path_external + f'{project_id}_{sample_type}_feature_intensities.csv', index=False)
sample_metadata.to_csv(output_path_external + f'{project_id}_{sample_type}_sample_metadata.csv', index=False)
print(feature_metadata.shape)
print(feature_intensities.shape)
print(sample_metadata.shape)

# COMMAND ----------

dbutils.fs.rm(output_path_external + 'feature_metadata.csv', True)
dbutils.fs.rm(output_path_external + 'feature_intensities.csv', True)
dbutils.fs.rm(output_path_external + 'sample_metadata.csv', True)

# COMMAND ----------



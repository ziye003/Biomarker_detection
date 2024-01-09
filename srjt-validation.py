# Databricks notebook source
token = dbutils.secrets.get(scope="zi.ye", key="github_token")

# COMMAND ----------

# MAGIC %pip install git+https://dangeloaguirre:$token@github.com/project-toolkit

# COMMAND ----------

from project_toolkit.qc import table_validation as tv
import pandas as pd
srjt = pd.read_excel('s3://hypoxia-pilot-study/01_raw_data/hypoxia-pilot-metabolomics_sample_run_join_table.xlsx')
srjt.dropna(inplace=True)
display(srjt)
validated_df = tv.SampleRunJoinTableValidator.run(srjt)

# COMMAND ----------



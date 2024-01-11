# Databricks notebook source
# MAGIC %pip install hydra-core

# COMMAND ----------

# Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

# COMMAND ----------

# Clear previous initialization
GlobalHydra.instance().clear()


with initialize(version_base=None,config_path="./config"):
    cfg = compose(config_name="config_misame3.yaml")

# COMMAND ----------

# MAGIC %md # linear regression

# COMMAND ----------

def perform_linear_regression(data, independent_variable, covariates, time_point):

    # independent_var = kwargs["independent_variable"]
    # covariates = kwargs["covariates"]
    # time_point = kwargs["time_point"]
    res_dict = { 'feature_label': data['feature_label'].iloc[0],'independent_variable': independent_variable}
    try:
        if data is not None and not data.empty:
            # Construct the linear regression formula
            formula = f"level ~ {' + '.join([cfg.analysis.covariate , independent_variable])}"

            # Fit the linear regression model
            model = sm.OLS.from_formula(formula, data=data)
            result = model.fit()

            # Extract p-value, effect size, and confidence intervals for the independent variable
            p_value = result.pvalues[independent_variable]
            effect_size = result.params[independent_variable]
            ci_lower = result.conf_int().loc[independent_variable, 0]
            ci_upper = result.conf_int().loc[independent_variable, 1]

            # Assign the results to the dictionary
            res_dict['LR_P_Value'] = p_value
            res_dict['LR_Effect_Size'] = effect_size
            res_dict['LR_CI_Lower'] = ci_lower
            res_dict['LR_CI_Upper'] = ci_upper

            return pd.DataFrame([res_dict])
        else:
            # Return an empty DataFrame if data is empty
            res_dict['LR_P_Value'] = np.nan
            res_dict['LR_Effect_Size'] = np.nan
            res_dict['LR_CI_Lower'] = np.nan
            res_dict['LR_CI_Upper'] = np.nan
            return pd.DataFrame([res_dict])
    except Exception as e:
        print(f'Error: {str(e)}')
        res_dict['LR_P_Value'] = np.nan
        res_dict['LR_Effect_Size'] = np.nan
        res_dict['LR_CI_Lower'] = np.nan
        res_dict['LR_CI_Upper'] = np.nan
        return pd.DataFrame([res_dict])

# COMMAND ----------

data_dt = spark.read.table(cfg.data.table_name)
mtb_dt = data_dt.select(*cfg.data.selected_columns).withColumnRenamed('metabolite', 'feature_label')
display(mtb_dt)

result_schema = StructType([
    StructField("feature_label", StringType(), True),
    StructField("independent_variable", StringType(), True),
    StructField("LR_P_Value", DoubleType(), True),
    StructField("LR_Effect_Size", DoubleType(), True),
    StructField("LR_CI_Lower", DoubleType(), True),
    StructField("LR_CI_Upper", DoubleType(), True),
])

# Loop over independent variables
for independent_variable in cfg.analysis.independent_variable_list:

    # Select relevant columns and filter the DataFrame
    reg_df = mtb_dt.select(col('level'), col('feature_label'), col(cfg.analysis.covariate), col('time_point'), col(independent_variable)).filter(
        (col(independent_variable) != 0) & (~col(independent_variable).isNull()) & (col('time_point') == cfg.analysis.time_point)
    )

    result_df = (
        reg_df
        .groupBy("feature_label")
        .applyInPandas(lambda df: perform_linear_regression(df, independent_variable, cfg.analysis.covariate, cfg.analysis.time_point), schema=result_schema)
        .withColumn("independent_variable", lit(independent_variable))
    )

    # Save the result DataFrame to a CSV file
    res_path = f'{cfg.analysis.result_path}/hydra_linear_regression_res_{cfg.analysis.time_point}_{independent_variable}.csv'
    result_df.toPandas().to_csv(res_path, index=False)

# COMMAND ----------

result_df.display()

# COMMAND ----------

distinct_subjects = data_dt.select(col("subject_id")).distinct().count()
distinct_subjects

# COMMAND ----------

distinct_sites = data_dt.select(col("csps_n")).distinct().count()
distinct_sites

# COMMAND ----------

# MAGIC %md # check

# COMMAND ----------

for independent_variable in cfg.analysis.independent_variable_list:


    res_path = f'{cfg.analysis.result_path}/hydra_linear_regression_res_{cfg.analysis.time_point}_{independent_variable}.csv'
    print(pd.read_csv(res_path).head())

# COMMAND ----------



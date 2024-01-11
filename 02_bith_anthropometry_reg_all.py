# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import pandas as pd
from pyspark.sql.functions import mean, stddev, col, split,isnan,lit

import statsmodels.api as sm
from scipy.stats import f_oneway, ttest_ind

import numpy as np

# COMMAND ----------

# MAGIC %sql
# MAGIC use imic_misame3

# COMMAND ----------

data_dt = spark.read.table('mtb_pheno_processed')
mtb_dt = data_dt.select('time_point', 'level', 'metabolite', 'treatment1', 'treatment2', 'code', 'csps_n', 'c_haz0', 'c_waz0', 'c_igu_hcaz0', 'c_igu_muacaz0', 'bmiz060')

mtb_dt = mtb_dt.withColumnRenamed('metabolite','feature_label')
display(mtb_dt)


# COMMAND ----------

# MAGIC %md # choose ind variable

# COMMAND ----------



# List of independent variables
INDEPENDENT_VARIABLE_LIST = ['c_haz0', 'c_waz0', 'c_igu_hcaz0', 'c_igu_muacaz0', 'bmiz060']
COVARIATES = ["csps_n"]
TIME_POINT = '1421'

# Define the schema for the result DataFrame
result_schema = StructType([
    StructField("feature_label", StringType(), True),
    StructField("independent_variable", StringType(), True),
    StructField("LR_P_Value", DoubleType(), True),
    StructField("LR_Effect_Size", DoubleType(), True),
    StructField("LR_CI_Lower", DoubleType(), True),
    StructField("LR_CI_Upper", DoubleType(), True),
])

# COMMAND ----------

# MAGIC %md # linear model
# MAGIC

# COMMAND ----------

# for INDEPENDENT_VARIABLE in INDEPENDENT_VARIABLE_LIST:
#     # Select relevant columns and filter the DataFrame
#     reg_df = mtb_dt.select(col('level'), col('feature_label'), col('csps_n'), col('time_point'), col(INDEPENDENT_VARIABLE)).filter(
#         (col(INDEPENDENT_VARIABLE) != 0) & (~col(INDEPENDENT_VARIABLE).isNull()) & (col('time_point') == TIME_POINT)
#     )

#     # Define a helper function for linear regression
#     def perform_linear_regression(data):
#         res_dict = {'independent_variable': INDEPENDENT_VARIABLE}
#         res_dict['feature_label'] = data['feature_label'].iloc
#         try:
#             if data is not None and not data.empty:
#                 # Construct the linear regression formula
#                 formula = f"level ~ {' + '.join(COVARIATES + [INDEPENDENT_VARIABLE])}"

#                 # Fit the linear regression model
#                 model = sm.OLS.from_formula(formula, data=data)
#                 result = model.fit()

#                 # Extract p-value, effect size, and confidence intervals for the independent variable
#                 p_value = result.pvalues[INDEPENDENT_VARIABLE]
#                 effect_size = result.params[INDEPENDENT_VARIABLE]
#                 ci_lower = result.conf_int().loc[INDEPENDENT_VARIABLE, 0]
#                 ci_upper = result.conf_int().loc[INDEPENDENT_VARIABLE, 1]

#                 # Assign the results to the dictionary
#                 res_dict['LR_P_Value'] = p_value
#                 res_dict['LR_Effect_Size'] = effect_size
#                 res_dict['LR_CI_Lower'] = ci_lower
#                 res_dict['LR_CI_Upper'] = ci_upper

#                 return pd.DataFrame([res_dict])
#             else:
#                 # Return an empty DataFrame if data is empty
#                 res_dict['LR_P_Value'] = np.nan
#                 res_dict['LR_Effect_Size'] = np.nan
#                 res_dict['LR_CI_Lower'] = np.nan
#                 res_dict['LR_CI_Upper'] = np.nan
#                 return pd.DataFrame([res_dict])
#         except Exception as e:
#             print(f'Error: {str(e)}')
#             res_dict['LR_P_Value'] = np.nan
#             res_dict['LR_Effect_Size'] = np.nan
#             res_dict['LR_CI_Lower'] = np.nan
#             res_dict['LR_CI_Upper'] = np.nan
#             return pd.DataFrame([res_dict])

#     # Apply linear regression using applyInPandas
#     result_df = (
#         reg_df
#         .groupBy("feature_label")
#         .applyInPandas(perform_linear_regression, schema=result_schema)
#         .withColumn("independent_variable", lit(INDEPENDENT_VARIABLE))
#     )

#     # Save the result DataFrame to a CSV file
#     res_path = f'/dbfs/mnt/client-002sap21p015-imic/04_data_analysis/results/linear_regression_res_{TIME_POINT}_{INDEPENDENT_VARIABLE}.csv'
#     result_df.toPandas().to_csv(res_path, index=False)

# COMMAND ----------

for INDEPENDENT_VARIABLE in INDEPENDENT_VARIABLE_LIST:
    # Select relevant columns and filter the DataFrame
    reg_df = mtb_dt.select(col('level'), col('feature_label'), col('csps_n'), col('time_point'), col(INDEPENDENT_VARIABLE)).filter(
        (col(INDEPENDENT_VARIABLE) != 0) & (~col(INDEPENDENT_VARIABLE).isNull()) & (col('time_point') == TIME_POINT)
    )

    # Define a helper function for linear regression
    def perform_linear_regression(data):
        res_dict = {'independent_variable': INDEPENDENT_VARIABLE}
        res_dict['feature_label'] = data['feature_label'].iloc[0]
        try:
            if data is not None and not data.empty:
                # Construct the linear regression formula
                formula = f"level ~ {' + '.join(COVARIATES + [INDEPENDENT_VARIABLE])}"

                # Fit the linear regression model
                model = sm.OLS.from_formula(formula, data=data)
                result = model.fit()

                # Extract p-value, effect size, and confidence intervals for the independent variable
                p_value = result.pvalues[INDEPENDENT_VARIABLE]
                effect_size = result.params[INDEPENDENT_VARIABLE]
                ci_lower = result.conf_int().loc[INDEPENDENT_VARIABLE, 0]
                ci_upper = result.conf_int().loc[INDEPENDENT_VARIABLE, 1]

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

    # Apply linear regression using applyInPandas
    result_df = (
        reg_df.groupBy("feature_label")
        .applyInPandas(lambda df: perform_linear_regression(df), schema=result_schema)
        .withColumn("independent_variable", lit(INDEPENDENT_VARIABLE))
    )

    # Save the result DataFrame to a CSV file
    res_path = f'/dbfs/mnt/client-002sap21p015-imic/04_data_analysis/results/linear_regression_res_{TIME_POINT}_{INDEPENDENT_VARIABLE}.csv'
    result_df.toPandas().to_csv(res_path, index=False)

# COMMAND ----------

# MAGIC %md #check

# COMMAND ----------

res_check = pd.read_csv(res_path)
display(res_check)

# COMMAND ----------



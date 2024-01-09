# Databricks notebook source
# COMMAND ----------

# MAGIC %run ../initialization/notebook_setup_plotting

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data preprocessings

# COMMAND ----------

with initialize(version_base=None, config_path="../config"):
    cfg = compose(config_name="config_hypoxia.yaml")
    hypoxia = hypoxia(cfg)  # noqa: F821

    hypoxia.generate_raw_grand_table(save_to_file=True)
    hypoxia.load_grand_table(type="raw")
    hypoxia.generate_long_format_data(spark=spark, raw_data=False, save_to_file=True)


# COMMAND ----------
# Check results

DataFileReader(
    file_path=hypoxia.cfg.files.grand_table_path + "grand_serum_preprocessed_long.delta"
).load(data_format="delta", spark=spark).display()

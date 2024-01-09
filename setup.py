# Databricks notebook source
import json

with open('../config/hypoxia-pilot-study/config.json') as project_config_json:
    project_config = json.load(project_config_json)

with open('../config/packaging/config.json') as packaging_config_json:
    packaging_config = json.load(packaging_config_json)

# COMMAND ----------

# MAGIC %r
# MAGIC options(
# MAGIC   repos = c(REPO_NAME = "https://packagemanager.rstudio.com/cran/__linux__/focal/latest"),
# MAGIC   HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"]))
# MAGIC )
# MAGIC
# MAGIC
# MAGIC install.packages("rjson")
# MAGIC project_config <- rjson::fromJSON(file="../config/hypoxia-pilot-study/config.json")

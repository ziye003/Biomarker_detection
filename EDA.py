# Databricks notebook source
# MAGIC %run ../initialization/notebook_setup_plotting

# COMMAND ----------

with initialize(version_base=None, config_path="../config"):
    cfg = compose(config_name="config_hypoxia.yaml")
    hypoxia = hypoxia(cfg)  # noqa: F821


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Duplicated samples

# COMMAND ----------

# Duplicated Samples
hypoxia.meta_data.pheno_combined.query('`Case ID` == "2701"')


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data aggregation

# COMMAND ----------

# Samples with matching metabolomics data
sample_ids = hypoxia.meta_data.sample_metadata['sample_id'].tolist()
sample_with_metab = hypoxia.meta_data.pheno_combined.query('`Custom ID` in @sample_ids')

# COMMAND ----------


def has_day0(visit):
    return 'Day 0' in list(visit)


def has_day168(visit):
    return 'Day 168' in list(visit)


def has_day336(visit):
    return 'Day 336' in list(visit)


sample_agg = (
    sample_with_metab.groupby(['Case ID', 'Visit'])
    .agg(count=('Visit', 'count'))
    .reset_index()
    .groupby('Case ID')
    .agg(
        count=('Visit', 'count'),
        has_day0=('Visit', has_day0),
        has_day168=('Visit', has_day168),
        has_day336=('Visit', has_day336),
    )
    .reset_index()
)

sample_agg

# COMMAND ----------

# Number of samples with complete timepoints (3 timepoints)
print(len(sample_agg.query('count == 3')))

# Number of cats
print(len(set(sample_with_metab['Case ID'])))

# COMMAND ----------

# breakdown
sample_with_metab.groupby(['Treatment', 'Progressive', 'Visit']).agg(
    visit_count=('Visit', 'count')
)

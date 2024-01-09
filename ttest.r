# Databricks notebook source
# MAGIC %run ../initialization/notebook_setup_plotting

# COMMAND ----------

library(reshape2)
library(magrittr)
library(dplyr)
library(purrr)
library(broom)
library(tidyr)
library(data.table)

# COMMAND ----------

data_long <- read.csv("/dbfs/mnt/hypoxia-pilot-study/04_data_analysis/grand_table/grand_serum_preprocessed_long.csv")
data_long$Visit <- gsub(" ", "", data_long$Visit)
# data_long = subset(data_long, not ((animal == "2701") & (Visit == "Day336")))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## CKD biomarkers associated with disease over time
# MAGIC > T-tests at each timepoint: Day0, Day168, Day336

# COMMAND ----------

data_long_placebo <- data_long %>% subset(Treatment == "T01")
data_long_placebo

# COMMAND ----------

perform_and_summarize_t_test <- function(data, timepoint) {
  data %>%
    subset(Visit == timepoint) %>%
    nest(data = -metabolite) %>%
    mutate(
      res = map(data, ~ t.test(level ~ Progressive, data = .)),
      tidied = map(res, tidy)
    ) %>%
    unnest(tidied) %>%
    select(
      metabolite, estimate, estimate1, estimate2,
      statistic, p.value, parameter, conf.low,
      conf.high, method, alternative
    )
}

timepoints <- c("Day0", "Day168", "Day336")

for (timepoint in timepoints) {
  print(paste("Timepoint =", timepoint))
  t_test_results <- perform_and_summarize_t_test(data_long_placebo, timepoint)

  res_file <- paste0("/dbfs/mnt/hypoxia-pilot-study/04_data_analysis/results/serum_placebo_ttest_", timepoint, ".csv")
  fwrite(t_test_results, res_file)
  print(paste("Results saved to", res_file))
}

# COMMAND ----------

# MAGIC %py
# MAGIC
# MAGIC # Check results
# MAGIC timepoints = ('Day0', 'Day168', 'Day336')
# MAGIC for timepoint in timepoints:
# MAGIC     print(f'Timepoint = {timepoint}')
# MAGIC     res_file = f'/dbfs/mnt/hypoxia-pilot-study/04_data_analysis/results/serum_placebo_ttest_{timepoint}.csv'
# MAGIC     res = pd.read_csv(res_file)
# MAGIC     display(res)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## CKD biomarkers changing in response to treatment
# MAGIC
# MAGIC > Paired T-tests
# MAGIC - baseline vs. midpoint,
# MAGIC - baseline vs. endpoint
# MAGIC
# MAGIC

# COMMAND ----------

data_long_treatment <- data_long %>% subset(Treatment == "T03")
data_long_treatment

# COMMAND ----------

data_wide <- data_long_treatment %>%
  pivot_wider(names_from = Visit, values_from = level, values_fill = 0) %>%
  select(metabolite, animal, Day0, Day168, Day336)

perform_and_summarize_paired_t_test <- function(data, timepoint) {
  nested <- data %>%
    nest(data = -metabolite) %>%
    mutate(
      res = map(data, ~ t.test(.[["Day0"]], .[[timepoint]], paired = TRUE)),
      tidied = map(res, tidy)
    )

  unnest(nested, tidied) %>%
    select(
      metabolite, estimate,
      statistic, p.value, parameter, conf.low,
      conf.high, method, alternative
    )
}

timepoints <- c("Day168", "Day336")

for (timepoint in timepoints) {
  paired_t_test_results <- perform_and_summarize_paired_t_test(data_wide, timepoint)
  res_file <- paste0("/dbfs/mnt/hypoxia-pilot-study/04_data_analysis/results/serum_treatment_paired_ttest_Day0vs", timepoint, ".csv")
  fwrite(paired_t_test_results, res_file)
  print(paste0("Results saved to", res_file))
}

# COMMAND ----------

# MAGIC %py
# MAGIC
# MAGIC # Check results
# MAGIC timepoints = ('Day168', 'Day336')
# MAGIC for timepoint in timepoints:
# MAGIC     print(f'Timepoint = {timepoint}')
# MAGIC     res_file = f'/dbfs/mnt/hypoxia-pilot-study/04_data_analysis/results/serum_treatment_paired_ttest_Day0vs{timepoint}.csv'
# MAGIC     res = pd.read_csv(res_file)
# MAGIC     display(res)


# COMMAND ----------
# MAGIC %md
# MAGIC
# MAGIC ## CKD biomarkers changing in response to treatment
# MAGIC
# MAGIC > Indepedent T-tests
# MAGIC - baseline vs. midpoint,
# MAGIC - baseline vs. endpoint
# MAGIC
# MAGIC

# COMMAND ----------

data_wide <- data_long_treatment %>%
  pivot_wider(names_from = Visit, values_from = level, values_fill = 0) %>%
  select(metabolite, animal, Day0, Day168, Day336)

perform_and_summarize_independent_t_test <- function(data, timepoint) {
  nested <- data %>%
    nest(data = -metabolite) %>%
    mutate(
      res = map(data, ~ t.test(.[["Day0"]], .[[timepoint]], paired = FALSE)),
      tidied = map(res, tidy)
    )

  unnest(nested, tidied) %>%
    select(
      metabolite, estimate,
      statistic, p.value, parameter, conf.low,
      conf.high, method, alternative
    )
}

timepoints <- c("Day168", "Day336")

for (timepoint in timepoints) {
  indep_t_test_results <- perform_and_summarize_independent_t_test(data_wide, timepoint)
  res_file <- paste0("/dbfs/mnt/hypoxia-pilot-study/04_data_analysis/results/serum_treatment_independent_ttest_Day0vs", timepoint, ".csv")
  fwrite(indep_t_test_results, res_file)
  print(paste0("Results saved to", res_file))
}



# COMMAND ----------

# MAGIC %py
# MAGIC
# MAGIC # Check results
# MAGIC timepoints = ('Day168', 'Day336')
# MAGIC for timepoint in timepoints:
# MAGIC     print(f'Timepoint = {timepoint}')
# MAGIC     res_file = f'/dbfs/mnt/hypoxia-pilot-study/04_data_analysis/results/serum_treatment_independent_ttest_Day0vs{timepoint}.csv'
# MAGIC     res = pd.read_csv(res_file)
# MAGIC     display(res)

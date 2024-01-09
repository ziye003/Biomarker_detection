# Databricks notebook source

# MAGIC %pip install pendulum

# COMMAND ----------

project_name = dbutils.widgets.get('project_name')

print(f'The current project is {project_name}')
# COMMAND ----------
import pendulum

print(f'the current day and time is {pendulum.now()}')
# COMMAND ----------

from src.example_module import subtract_numbers

subtraction = subtract_numbers(10, 4)

# COMMAND ----------

print(subtraction)

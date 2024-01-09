import os
import re
from typing import Optional, Union

import awswrangler as wr
import pandas as pd
from pydantic import BaseModel
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import element_at, input_file_name, split


class DataFileReader(BaseModel):
    file_path: str
    dataset: Optional[Union[pd.DataFrame, SparkDataFrame]] = None

    class Config:
        arbitrary_types_allowed = True

    def _read_data(self, use_awswrangler=False, dtype={}):
        try:
            if use_awswrangler:
                return wr.s3.read_csv(self.file_path)

            chunks = pd.read_csv(
                self.file_path,
                low_memory=False,
                iterator=True,
                chunksize=1000,
                dtype=dtype,
            )
            return pd.concat(chunks)
        except Exception as error:
            print(
                f"Error occurred while reading data from path: {self.file_path}\nError message: {error}"
            )
            return None

    @property
    def has_wildcard_in_file_path(self):
        pattern = r"[\*\?\[\]\{\}]"
        return bool(re.search(pattern, self.file_path))

    @property
    def is_directory(self):
        file_name = self.file_path.split("/")[-1]
        return not bool(re.search(r"\.", file_name))

    def _read_data_spark(self, data_format, delimiter, spark):
        file_path = self._convert_to_spark_path()

        res = (
            spark.read.format(data_format)
            .option("header", "true")
            .option("inferSchema", "true")
            .option("delimiter", delimiter)
            .load(file_path)
        )

        if self.has_wildcard_in_file_path:
            res = res.withColumn("Source", input_file_name()).withColumn(
                "Source", element_at(split("Source", "/"), -1)
            )

        return res

    def _read_data_table(self, spark):
        return spark.read.table(self.file_path)

    def _convert_to_spark_path(self) -> str:
        if self.file_path.startswith("/dbfs"):
            return "dbfs:" + self.file_path[5:]
        return self.file_path

    def _load_with_spark(self, data_format, delimiter, spark):
        is_s3_path = any(keyword in self.file_path for keyword in ("dbfs:", "s3:"))
        is_delta_file = any(keyword in self.file_path for keyword in ("delta"))

        if (
            is_s3_path
            or self.has_wildcard_in_file_path
            or self.is_directory
            or is_delta_file
        ):
            return self._read_data_spark(data_format, delimiter, spark)

    def _load_without_spark(self):
        is_s3_path = any(keyword in self.file_path for keyword in ("dbfs:", "s3:"))
        is_plain_txt = any(keyword in self.file_path for keyword in ("csv", "txt"))

        if is_plain_txt and is_s3_path:
            return self._read_data(use_awswrangler=True)
        if is_plain_txt:
            return self._read_data()

    def load(self, data_format="csv", delimiter=",", spark=None):
        print(f"Loading data from {self.file_path} ... ")

        if data_format == "table":
            return self._read_data_table(spark)

        if spark:
            return self._load_with_spark(data_format, delimiter, spark)
        else:
            return self._load_without_spark()


def convert_to_spark_path(path: str) -> str:
    if path.startswith("/dbfs"):
        return "dbfs:" + path[5:]
    else:
        return path


class DataFileWriter:
    @staticmethod
    def save2files(res, file_path):
        path = os.path.dirname(file_path)
        os.makedirs(path, exist_ok=True)
        res.to_csv(file_path, index=False)
        print(f"Results saved to: {file_path}")

    @staticmethod
    def save2files_spark(res, file_path, spark):
        path = os.path.dirname(file_path)
        os.makedirs(path, exist_ok=True)
        spark.createDataFrame(res).write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(convert_to_spark_path(file_path))
        print(f"Results saved to: {file_path}")

    @staticmethod
    def save_to_tables(res, table_name, file_path, spark):
        path = os.path.dirname(file_path)
        os.makedirs(path, exist_ok=True)

        database_name = table_name.split(".")[0]
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")

        if isinstance(res, pd.DataFrame):
            spark.createDataFrame(res).write.format("delta").saveAsTable(
                table_name, mode="overwrite", path=convert_to_spark_path(file_path)
            )
        else:
            res.write.format("delta").saveAsTable(
                table_name, mode="overwrite", path=convert_to_spark_path(file_path)
            )

        print(f"The following delta table has been generated: {table_name}")
        print(f"Results saved to: {file_path}")

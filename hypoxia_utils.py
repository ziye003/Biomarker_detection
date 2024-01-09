from collections import namedtuple

# import numpy as np
import pandas as pd

from toolkits.data_utils import DataFileReader, DataFileWriter
from toolkits.preprocessing_utils import Preprocessing

MetabolomicsData = namedtuple("MetabolomicsData", ["serum"])
CytokineData = namedtuple("CytokineData", ["serum"])

MetabolomicsDataPreprocessed = namedtuple("MetabolomicsDataPreprocessed", ["serum"])
CytokineDataPreprocessed = namedtuple("CytokineDataPreprocessed", ["serum"])
GrandTable = namedtuple("GrandTable", ["serum"])
MetaData = namedtuple(
    "MetaData", ['pheno_data', 'manifest_data', 'pheno_combined', 'sample_metadata']
)


class Zoetis_cytokine(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._load_meta_data()
        # self._load_cytokine_data()

    def _load_meta_data(self):
        pheno_data = pd.read_csv(
            self.cfg.files.pheno,
            dtype={"animal": str},
        ).drop_duplicates()

        manifest_data = pd.read_csv(
            self.cfg.files.manifest,
            dtype={"Case ID": str},
        )

        pheno_combined = pheno_data.merge(
            manifest_data,
            left_on=["animal", 'visit (DCF3494)', 'Treatment'],
            right_on=['Case ID', 'Visit', 'Treatment ID'],
            how="outer",
        )

        sample_metadata = pd.read_csv(
            self.cfg.files.sample_metadata,
        )

        self.meta_data = MetaData(
            pheno_data, manifest_data, pheno_combined, sample_metadata
        )

    def _load_cytokine_data(self, selected_mtb=None):
        cytokine_data = (
            DataFileReader(file_path=self.cfg.files.cytokine.serum)
            .load(data_format="csv")
            .T.reset_index()
        )
        cytokine_data.columns = 'serum_' + cytokine_data.iloc[0]
        cytokine_data = cytokine_data.iloc[1:].rename(
            columns={"serum_targetName": "sample_id"}
        )
        self.cytokine_data = CytokineData(cytokine_data)

    def generate_raw_grand_table(self, save_to_file=False):
        self._load_cytokine_data()

        self.cytokine_data.serum["sample_id"] = (
            self.cytokine_data.serum["sample_id"]
            .astype(str)
            .str.strip()
            .str.replace(" ", "")
        )

        grand_table = self.cytokine_data.serum.merge(
            self.meta_data.pheno_combined,
            left_on="sample_id",
            right_on="Custom ID",
            how="inner",
        )

        self.grand_table = GrandTable(grand_table)
        if save_to_file:
            DataFileWriter.save2files(
                grand_table,
                self.cfg.files.grand_table_path + "grand_cytokine_serum_" + "raw.csv",
            )

    def load_grand_table(self, type="raw"):
        grand_table = DataFileReader(
            file_path=self.cfg.files.grand_table_path
            + "grand_cytokine_serum_"
            + f"{type}.csv"
        ).load(data_format="csv")

        self.grand_table = GrandTable(grand_table)

    def _generate_preprocessed_grand_table(self, save_to_file=False):
        cytokine_data_grand_preprocessed = []

        for sample_type in self.grand_table._fields:
            cytokine_string = sample_type.replace("_", "")

            preprocess = Preprocessing(
                getattr(self.grand_table, sample_type),
                mtb_string=cytokine_string,
            )

            cytokine_preprocessed = preprocess.run(
                steps=['scale'],
            )

            if save_to_file:
                DataFileWriter.save2files(
                    cytokine_preprocessed,
                    self.cfg.files.grand_table_path
                    + f"grand_cytokine_{sample_type}_preprocessed.csv",
                )

            cytokine_data_grand_preprocessed.append(cytokine_preprocessed)

        self.cytokine_data_grand_preprocessed = CytokineDataPreprocessed(
            *cytokine_data_grand_preprocessed,
        )

    def generate_long_format_data(self, spark, raw_data=False, save_to_file=False):
        if raw_data:
            self.load_grand_table(type="raw")
            grand_table = self.grand_table
        else:
            self._generate_preprocessed_grand_table(save_to_file=True)
            grand_table = self.cytokine_data_grand_preprocessed

        id_vars_base = self.cfg.wide2long.id_vars_base

        for sample_type in grand_table._fields:
            cytokine_grand_table = getattr(grand_table, sample_type)
            cytokine_string = sample_type.replace("_", "")
            cytokine_names = [
                col for col in cytokine_grand_table.columns if cytokine_string in col
            ]
            id_vars = id_vars_base.copy()
            data_long = cytokine_grand_table.melt(
                id_vars=id_vars,
                value_vars=cytokine_names,
                var_name="cytokine",
                value_name="level",
            )

            file_path = f"{self.cfg.files.grand_table_path}grand_cytokine_{sample_type}{'_raw_long' if raw_data else '_preprocessed_long'}.delta"

            if save_to_file:
                DataFileWriter.save2files_spark(data_long, file_path, spark)


class Zoetis(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._load_meta_data()
        self._load_cytokine_data()
        # self._load_metabolomics_data()
        # self.metabolomics_data_preprocessed = MetabolomicsDataPreprocessed(None)
        # self.grand_table = GrandTable(None)

    def _load_meta_data(self):
        pheno_data = pd.read_csv(
            self.cfg.files.pheno,
            dtype={"animal": str},
        ).drop_duplicates()

        manifest_data = pd.read_csv(
            self.cfg.files.manifest,
            dtype={"Case ID": str},
        )

        pheno_combined = pheno_data.merge(
            manifest_data,
            left_on=["animal", 'visit (DCF3494)', 'Treatment'],
            right_on=['Case ID', 'Visit', 'Treatment ID'],
            how="outer",
        )

        sample_metadata = pd.read_csv(
            self.cfg.files.sample_metadata,
        )

        self.meta_data = MetaData(
            pheno_data, manifest_data, pheno_combined, sample_metadata
        )

    def _load_metabolomics_data(self, selected_mtb=None):
        metab_data = DataFileReader(file_path=self.cfg.files.metab.serum).load(
            data_format="csv"
        )
        self.metabolomics_data = MetabolomicsData(metab_data)

    def generate_raw_grand_table(self, save_to_file=False):
        self._load_metabolomics_data()

        self.metabolomics_data.serum["sample_id"] = (
            self.metabolomics_data.serum["sample_id"]
            .astype(str)
            .str.strip()
            .str.replace(" ", "")
        )

        grand_table = self.metabolomics_data.serum.merge(
            self.meta_data.pheno_combined,
            left_on="sample_id",
            right_on="Custom ID",
            how="inner",
        )

        self.grand_table = GrandTable(grand_table)
        if save_to_file:
            DataFileWriter.save2files(
                grand_table,
                self.cfg.files.grand_table_path + "grand_serum_" + "raw.csv",
            )

    def load_grand_table(self, type="raw"):
        grand_table = DataFileReader(
            file_path=self.cfg.files.grand_table_path + "grand_serum_" + f"{type}.csv"
        ).load(data_format="csv")

        self.grand_table = GrandTable(grand_table)

    def _generate_preprocessed_grand_table(self, save_to_file=False):
        metabolomics_data_grand_preprocessed = []

        for sample_type in self.grand_table._fields:
            mtb_string = sample_type.replace("_", "")

            preprocess = Preprocessing(
                getattr(self.grand_table, sample_type),
                mtb_string=mtb_string,
            )

            metab_preprocessed = preprocess.run(
                steps=self.cfg.preprocessing.steps,
                impute_strategy=self.cfg.preprocessing.fill_na.impute_strategy,
                dropping_thres=self.cfg.preprocessing.drop_na.dropping_thres,
            )
            if save_to_file:
                DataFileWriter.save2files(
                    metab_preprocessed,
                    self.cfg.files.grand_table_path
                    + f"{sample_type}_grand_preprocessed.csv",
                )

            metabolomics_data_grand_preprocessed.append(metab_preprocessed)

        self.metabolomics_data_grand_preprocessed = MetabolomicsDataPreprocessed(
            *metabolomics_data_grand_preprocessed,
        )

    def generate_long_format_data(self, spark, raw_data=False, save_to_file=False):
        self._generate_preprocessed_grand_table(save_to_file=True)
        for sample_type in self.metabolomics_data_grand_preprocessed._fields:
            grand_table = getattr(
                self.metabolomics_data_grand_preprocessed, sample_type
            )
            mtb_string = sample_type.replace("_", "")
            mtb_names = [col for col in grand_table.columns if mtb_string in col]

            id_vars_base = self.cfg.wide2long.id_vars_base

            data_long = grand_table.melt(
                id_vars=id_vars_base,
                value_vars=mtb_names,
                var_name="metabolite",
                value_name="level",
            )

            file_path = f"{self.cfg.files.grand_table_path}"
            file_path += f"grand_{sample_type}"
            file_path += "_long.delta" if raw_data else "_preprocessed_long.delta"

            if save_to_file:
                DataFileWriter.save2files_spark(data_long, file_path, spark)

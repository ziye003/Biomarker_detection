import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, df, mtb_names=None, mtb_string=None):
        self.df = df.reset_index().iloc[:, 1:]
        self.mtb_string = mtb_string
        self.mtb_names = mtb_names or [
            col for col in self.df.columns if mtb_string in col
        ]
        self.mtb_names = np.array(self.mtb_names)

    def _separate_mtb_and_other_cols(self):
        df_mtb = self.df.loc[:, self.mtb_names]
        df_others = self.df.loc[
            :, [c for c in self.df.columns if c not in self.mtb_names]
        ]
        return df_mtb, df_others

    def replace_zero_with_na(self):
        print("> Replacing zero with NA ...")
        df_mtb, df_others = self._separate_mtb_and_other_cols()
        df_mtb.replace(0, np.nan, inplace=True)
        self.df = pd.concat([df_others, df_mtb], axis=1)

    def drop_na(self, dropping_thres):
        print("> Dropping data by NA rates ...")
        df_mtb, df_others = self._separate_mtb_and_other_cols()
        to_keep_bool = (df_mtb.isnull().sum(axis=0) / df_mtb.shape[0]) < dropping_thres
        df_mtb = df_mtb.loc[:, df_mtb.columns[to_keep_bool]]

        if not self.mtb_string:
            self.mtb_names = [c for c in df_mtb.columns if c in self.mtb_names]
        else:
            self.mtb_names = [c for c in df_mtb.columns if self.mtb_string in c]

        self.df = pd.concat([df_others, df_mtb], axis=1)

        n_dropped_columns = len(to_keep_bool) - sum(to_keep_bool)
        percentage_dropped = round(100 * n_dropped_columns / len(to_keep_bool), 1)

        message = f"{n_dropped_columns} ({percentage_dropped}%) columns were dropped due to high missingness (thres = {dropping_thres})!"
        print(message)

    def fill_na(self, strategy, drop_empty_columns=True):
        print(f"> Filling NAs using {strategy} strategy ...")
        df_mtb, df_others = self._separate_mtb_and_other_cols()
        if strategy == "min":
            df_mtb = df_mtb.fillna(df_mtb.min())
        elif strategy == "zero":
            df_mtb = df_mtb.fillna(0)
        elif strategy == "uniform":
            np.random.seed(seed=10)
            uniform_df = pd.DataFrame(
                [
                    np.random.uniform(minV / 10, minV, df_mtb.shape[0])
                    if not np.isnan(minV)
                    else np.array([np.nan] * df_mtb.shape[0])
                    for minV in df_mtb.min().values
                ],
                columns=range(df_mtb.shape[0]),
                index=df_mtb.columns,
            ).T
            uniform_df.update(df_mtb)
            df_mtb = uniform_df

        else:
            raise NotImplementedError(
                "The requested imputation method has not been implemented yet!"
            )
        if drop_empty_columns:
            is_all_empty = df_mtb.isna().all()
            if is_all_empty.sum() > 0:
                df_mtb = df_mtb.loc[:, ~is_all_empty]
                self.mtb_names = [c for c in df_mtb.columns if c in self.mtb_names]
                print(f"{is_all_empty.sum()} empty columns were dropped.")

        self.df = pd.concat([df_others, df_mtb], axis=1)

    def log_transformation(self):
        print("> Log transformation ...")
        df_mtb, df_others = self._separate_mtb_and_other_cols()
        df_mtb.replace(0, 1e-200, inplace=True)
        df_mtb_log = np.log(df_mtb)
        self.df = pd.concat(
            [df_others, df_mtb_log],
            axis=1,
        )

    def standardization(self):
        print("> Standarization...")
        df_mtb, df_others = self._separate_mtb_and_other_cols()
        df_mtb = pd.DataFrame(
            StandardScaler().fit_transform(df_mtb), columns=df_mtb.columns
        )
        self.df = pd.concat([df_others, df_mtb], axis=1)

    def run(self, steps=["scale"], impute_strategy="uniform", dropping_thres=0.9):
        func_mapper = {
            "replace_zero": lambda: self.replace_zero_with_na(),
            "drop_na": lambda: self.drop_na(dropping_thres),
            "fill_na": lambda: self.fill_na(strategy=impute_strategy),
            "log": lambda: self.log_transformation(),
            "scale": lambda: self.standardization(),
        }

        for step in steps:
            func_mapper.get(
                step, lambda: print("The preprocessing function is not defined!!!")
            )()

        print("Data preprocessing done!")
        return self.df

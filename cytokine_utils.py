import numpy as np
import pandas as pd


class OutlierDetector:
    """
    A class for processing data and performing outlier detection.

    Parameters:
    - data (pandas.DataFrame): The input data to be processed.
    - index_col (str): The column name to be used as the index (default: "TargetName").

    Methods:
    - get_abs_mi_value(): Calculates the absolute median deviation of the data.
    - mask_outliers(threshold): Masks out the outliers in the data based on a threshold.

    """

    def __init__(self, data, index_col="TargetName"):
        self.data = data.set_index(index_col)
        self.index_col = index_col

    def get_abs_mi_value(self):
        """
        Calculates the absolute Mi values of the data.

        Returns:
        - abs_mi (pandas.DataFrame): The absolute median deviation of the data.

        """
        mad = self._calculate_median_absolute_deviation()
        abs_mi = np.abs(
            0.6745 * (self.data - np.median(self.data, axis=1).reshape(-1, 1)) / mad
        )
        return abs_mi

    def _calculate_median_absolute_deviation(self):
        """
        Calculates the median absolute deviation (MAD) of the data.

        Returns:
        - mad (pandas.DataFrame): The median absolute deviation of the data.

        """
        return (
            np.median(
                np.absolute(self.data - np.median(self.data, axis=1).reshape(-1, 1)),
                axis=1,
            ).reshape(-1, 1)
            + 1e-200
        )

    def mask_outliers(self, threshold=2):
        """
        Masks out the outliers in the data based on a threshold.

        Parameters:
        - threshold (float): The threshold value for outlier detection (default: 2).

        Returns:
        - data_masked (pandas.DataFrame): The data with outliers masked.

        """
        abs_mi = self.get_abs_mi_value()
        has_outlier = np.sum(abs_mi > threshold, axis=1) > 0
        data_with_outlier = self.data.loc[has_outlier, :]

        row_max_idx = abs_mi.idxmax(axis=1)
        mask = abs_mi.apply(lambda x: x.name == row_max_idx[x.index], axis=0)
        data_with_outlier = data_with_outlier.mask(mask, other=np.nan)

        data_masked = pd.concat(
            [self.data.loc[~has_outlier, :], data_with_outlier],
            axis=0,
            ignore_index=False,
        )
        return data_masked


class LODCalculator:
    """
    A class that calculates the Limit of Detection (LOD) using data processing techniques.

    Attributes:
        data_processor (DataProcessor): An instance of the DataProcessor class used for data processing.

    Methods:
        get_LOD(singleplex=False): Calculates the LOD based on the masked data.

    """

    def __init__(self, outlier_detector):
        self.outlier_detector = outlier_detector

    def get_LOD(self, singleplex=False):
        """
        Calculates the Limit of Detection (LOD) based on the masked data.

        Args:
            singleplex (bool, optional): Specifies whether the LOD calculation is for singleplex data.
                Defaults to False.

        Returns:
            numpy.ndarray: An array containing the calculated LOD values.

        """
        data_masked = self.outlier_detector.mask_outliers(threshold=2)
        multiplier = 2.5 if singleplex else 3
        return np.mean(data_masked, axis=1) + multiplier * np.std(data_masked, axis=1)


class LODCorrector:
    """
    A class for performing LOD correction on data.

    Attributes:
        data (pandas.DataFrame): The input data.
        LOD (float): The LOD (Limit of Detection) value.
        index_col (str): The name of the column to be used as the index.

    Methods:
        get_LOD_corrected_data: Returns the LOD-corrected data.
    """

    def __init__(self, data, LOD, index_col="targetName"):
        self.data = data.set_index(index_col)
        self.LOD = LOD
        self.index_col = index_col

    def get_LOD_corrected_data(self, strategy="replace_with_LOD"):
        """
        Returns the LOD-corrected data.

        Returns:
            pandas.DataFrame: The LOD-corrected data.
        """
        EXCEPTIONS = ["KNG1", "CRP"]

        mask0 = self.data.eq(0, axis=0)
        mask = self.data.lt(self.LOD, axis=0)
        other = self.LOD if strategy == 'replace_with_LOD' else np.nan
        LOD_corrected_data = self.data.mask(mask, other=other, axis=0)

        LOD_corrected_data.loc[
            LOD_corrected_data.index.isin(EXCEPTIONS), :
        ] = self.data.loc[self.data.index.isin(EXCEPTIONS), :]

        cytokines_to_drop = (
            mask[mask.all(axis=1)].index.tolist()
            + mask0[mask0.all(axis=1)].index.tolist()
        )
        LOD_corrected_data.drop(cytokines_to_drop, inplace=True)

        LOD_corrected_data = LOD_corrected_data.reset_index().rename(
            columns={"index": self.index_col}
        )

        return LOD_corrected_data

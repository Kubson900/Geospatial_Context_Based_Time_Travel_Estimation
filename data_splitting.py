import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split


def train_dev_test_split_bucket_trajectory(
    gdf: gpd.GeoDataFrame,
    target_column: str,
    test_size: float = 0.2,
    bucket_number: int = 3,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Method to generate train/dev/test split from GeoDataFrame stratified by target_column.

    Args:
        gdf (gpd.GeoDataFrame): input GeoDataFrame
        target_column (str): Column to stratify on
        test_size (float): Percentage of test set
        bucket_number (int): Number of bins used to stratify data

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: Train, Dev, Test splits in GeoDataFrames
    """

    def calculate_trajectory_duration(df: pd.DataFrame) -> float:
        """
        Calculate the duration of a trajectory based on timestamps in a DataFrame.

        Args:
        df (pandas.DataFrame): A DataFrame containing a column 'timestamp'
                               with datetime objects.

        Returns:
        float: The duration of the trajectory in seconds.
        """
        if df.empty or df["timestamp"].nunique() == 1:
            return 0.0

        min_time = df["timestamp"].min()
        max_time = df["timestamp"].max()
        return float((max_time - min_time).total_seconds())

    gdf_copy = gdf.copy()

    trajectory_durations = (
        gdf_copy.groupby(target_column)
        .apply(calculate_trajectory_duration)
        .reset_index(name="duration")
    )
    gdf_copy_duration = gdf_copy.merge(trajectory_durations, on=target_column)
    gdf_copy_duration["duration_bin"] = pd.cut(
        gdf_copy_duration["duration"], bins=bucket_number, labels=False
    )
    trajectory_indices = gdf_copy_duration[target_column].unique()
    duration_bins = (
        gdf_copy_duration[[target_column, "duration_bin"]]
        .drop_duplicates()
        .set_index(target_column)["duration_bin"]
    )

    train_indices, test_indices = train_test_split(
        trajectory_indices,
        test_size=test_size * 2,
        stratify=duration_bins.loc[trajectory_indices],
    )
    dev_indices, test_indices = train_test_split(
        test_indices, test_size=0.5, stratify=duration_bins.loc[test_indices]
    )

    train_gdf = gdf_copy[gdf_copy[target_column].isin(train_indices)]
    dev_gdf = gdf_copy[gdf_copy[target_column].isin(dev_indices)]
    test_gdf = gdf_copy[gdf_copy[target_column].isin(test_indices)]

    return train_gdf, dev_gdf, test_gdf

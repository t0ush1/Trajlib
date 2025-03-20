import numpy as np
import pandas as pd


def read_data_chengdu(filepath, nrows):
    data = pd.read_csv(
        filepath,
        names=["driver_id", "order_id", "timestamp", "lon", "lat"],
        nrows=nrows,
    )
    trajectories = []
    for traj_id, group in data.groupby("order_id"):
        group = group.sort_values(by="timestamp")
        coordinates = group[["lon", "lat"]].values.tolist()
        timestamps = group["timestamp"].values.tolist()
        trajectories.append((coordinates, timestamps))
    return trajectories

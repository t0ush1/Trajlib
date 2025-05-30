import numpy as np
import pandas as pd
import transbigdata as tbd
from coord_convert import transform


def read_data_chengdu(filepath, nrows, clean=False, to_gps=False):
    data = pd.read_csv(
        filepath,
        names=["driver_id", "order_id", "timestamp", "lon", "lat"],
        nrows=nrows,
    )

    if clean:
        iter_clean = 1
        dislimit = 500
        anglelimit = 45
        speedlimit = 180
        method = "oneside"
        for i in range(iter_clean):
            data = tbd.traj_clean_drift(
                data=data,
                col=["order_id", "timestamp", "lon", "lat"],
                dislimit=dislimit,
                anglelimit=anglelimit,
                speedlimit=speedlimit,
                method=method,
            )

    trajectories = []
    for traj_id, group in data.groupby("order_id"):
        group = group.sort_values(by="timestamp")
        coordinates = group[["lon", "lat"]].values.tolist()
        timestamps = group["timestamp"].values.tolist()

        if to_gps:
            coordinates = [transform.gcj2wgs(lon, lat) for lon, lat in coordinates]

        trajectories.append((coordinates, timestamps))
    return trajectories

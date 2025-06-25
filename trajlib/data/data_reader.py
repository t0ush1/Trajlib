import numpy as np
import pandas as pd
import transbigdata as tbd
from coord_convert import transform


def read_data_chengdu(filepath, nrows, clean=False):
    data = pd.read_csv(
        filepath,
        names=["driver_id", "order_id", "timestamp", "lon", "lat"],
        nrows=nrows,
        header=None,
        dtype={"driver_id": str, "order_id": str},
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
        coordinates = [transform.gcj2wgs(lon, lat) for lon, lat in coordinates]
        timestamps = group["timestamp"].values.tolist()
        trajectories.append((coordinates, timestamps, {}))
    return trajectories


def read_data_geolife(filepath, nrows):
    data = pd.read_csv(
        filepath,
        names=["lat", "lon", "traj_id", "user_id", "mode", "timestamp"],
        nrows=nrows,
        header=None,
        dtype={"traj_id": str, "user_id": str},
    )

    trajectories = []
    for traj_id, group in data.groupby("traj_id"):
        group = group.sort_values(by="timestamp")
        mode_changes = (group["mode"] != group["mode"].shift()).cumsum()
        for _, sub_group in group.groupby(mode_changes):
            coordinates = sub_group[["lon", "lat"]].values.tolist()
            timestamps = sub_group["timestamp"].values.tolist()
            mode = sub_group["mode"].iloc[0]
            trajectories.append((coordinates, timestamps, {"mode": mode}))
    return trajectories


def read_data_bj(filepath, nrows):
    data = pd.read_csv(
        filepath,
        names=["traj_id", "user_id", "geo_id", "lon", "lat", "timestamp", "vflag"],
        nrows=nrows,
        header=None,
        dtype={"traj_id": str, "user_id": str},
    )

    trajectories = []
    for traj_id, group in data.groupby("traj_id"):
        group = group.sort_values(by="timestamp")
        coordinates = group[["lon", "lat"]].values.tolist()
        timestamps = group["timestamp"].values.tolist()
        vflag = group["vflag"].iloc[0]
        trajectories.append((coordinates, timestamps, {"vflag": vflag}))
    return trajectories

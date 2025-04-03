import numpy as np
import pandas as pd
import transbigdata as tbd
from coord_convert.transform import gcj2wgs

def read_data_chengdu(filepath, nrows, data_form):
    data = pd.read_csv(
        filepath,
        names=["driver_id", "order_id", "timestamp", "lon", "lat"],
        nrows=nrows,
    )
    data["timestamp"] = data["timestamp"] * 1000000000
    
    if data_form=="roadnet":
        # 清洗数据
        iter_clean=1,
        dislimit=500,
        anglelimit=45,
        speedlimit=180,
        method="oneside",
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
        if data_form=="roadnet":
            # 坐标转换
            new_coord = []
            for point in coordinates:
                new_lon, new_lat = gcj2wgs(point[0], point[1])
                new_coord.append([new_lon, new_lat])
        timestamps = group["timestamp"].values.tolist()
        trajectories.append((new_coord, timestamps))
    return trajectories
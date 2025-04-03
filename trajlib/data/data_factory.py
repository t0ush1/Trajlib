from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem

from trajlib.data.data_reader.chengdu import read_data_chengdu
from trajlib.data.data import GPSTrajData, GridTrajData, GridGraphData, RoadNetTrajData, RoadNetGraphData
from trajlib.data.RoadNetSystem import RoadNetSystem
import numpy as np

def build_grid(coord_trajs, step=100):
    all_lons = [p[0] for t in coord_trajs for p in t]
    all_lats = [p[1] for t in coord_trajs for p in t]
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)

    boundary_original = trajdl_cpp.RectangleBoundary(min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat)
    boundary = boundary_original.to_web_mercator()

    grid = SimpleGridSystem(boundary, step_x=step, step_y=step)
    return grid


def create_data(config):
    data_config = config["data_config"]

    if data_config["data_name"] == "chengdu":
        raw_data = read_data_chengdu(data_config["data_path"], data_config["data_size"], data_config["data_form"])
        # print(raw_data)

    if data_config["data_form"] == "gps":
        traj_data = GPSTrajData(raw_data)
        graph_data = None
    elif data_config["data_form"] == "grid":
        grid = build_grid([traj[0] for traj in raw_data], step=data_config["grid_step"])
        traj_data = GridTrajData(raw_data, grid)
        graph_data = GridGraphData(grid)
        config["model_config"]["vocab_size"] = len(grid)
    elif data_config["data_form"] == "roadnet":
        all_points = np.array([point for trajectory in raw_data for point in trajectory[0]])
        lons = all_points[:, 0]
        lats = all_points[:, 1]
        bounds=[
            lons.min(), # lon:left
            lats.min(), # lat:bottom
            lons.max(), # lon:right
            lats.max(), # lat:top
        ]
        # print(bounds)
        road_net = RoadNetSystem(bounds=bounds, cache_dir="./", network_type="all")
        traj_data = RoadNetTrajData(raw_data, road_net)
        graph_data = RoadNetGraphData(road_net)
        config["model_config"]["vocab_size"] = road_net.edge_num

    return traj_data, graph_data

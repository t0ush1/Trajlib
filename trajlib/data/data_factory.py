import os
from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem
import pickle

from trajlib.data.data_reader import read_data_chengdu, read_data_geolife, read_data_bj
from trajlib.data.data import TrajData, GridGraphData, RoadnetGraphData
from trajlib.data.roadnet_system import RoadnetSystem


def get_bounds(coord_trajs, loose=100):
    all_lons = [p[0] for t in coord_trajs for p in t]
    all_lats = [p[1] for t in coord_trajs for p in t]
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    lpoint = trajdl_cpp.convert_gps_to_webmercator(min_lon, min_lat)
    rpoint = trajdl_cpp.convert_gps_to_webmercator(max_lon, max_lat)
    lx, ly = lpoint.x - loose, lpoint.y - loose
    rx, ry = rpoint.x + loose, rpoint.y + loose
    lpoint = trajdl_cpp.convert_webmercator_to_gps(lx, ly)
    rpoint = trajdl_cpp.convert_webmercator_to_gps(rx, ry)
    return lpoint.lng, lpoint.lat, rpoint.lng, rpoint.lat


def build_grid(coord_trajs, step):
    min_lon, min_lat, max_lon, max_lat = get_bounds(coord_trajs)
    boundary_original = trajdl_cpp.RectangleBoundary(min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat)
    boundary = boundary_original.to_web_mercator()
    grid = SimpleGridSystem(boundary, step_x=step, step_y=step)
    return grid


def build_roadnet(coord_trajs, road_type):
    return RoadnetSystem(bounds=get_bounds(coord_trajs), cache_dir="./resource/osm_cache", network_type=road_type)


def create_data(config, overwrite=False):
    data_config = config["data_config"]

    cache_path = data_config["cache_path"]
    if overwrite or not os.path.exists(cache_path):
        data_path = data_config["data_path"]
        data_size = data_config["data_size"]

        match data_config:
            case {"data_name": "chengdu"}:
                raw_data = read_data_chengdu(data_path, data_size)
            case {"data_name": "geolife"}:
                raw_data = read_data_geolife(data_path, data_size)
            case {"data_name": "bj"}:
                raw_data = read_data_bj(data_path, data_size)
            case _:
                raise ValueError()

        grid = build_grid([traj[0] for traj in raw_data], data_config["grid_step"])
        roadnet = build_roadnet([traj[0] for traj in raw_data], data_config["road_type"])
        traj_data = TrajData(raw_data, data_config["window"], grid, roadnet)
        grid_graph_data = GridGraphData(grid)
        road_graph_data = RoadnetGraphData(roadnet)

        with open(cache_path, "wb") as f:
            pickle.dump((traj_data, grid_graph_data, road_graph_data), f)

    return load_data(config)


def load_data(config) -> tuple[TrajData, GridGraphData, RoadnetGraphData]:
    cache_path = config["data_config"]["cache_path"]
    with open(cache_path, "rb") as f:
        traj_data, grid_graph_data, road_graph_data = pickle.load(f)

    embedding_config = config["embedding_config"]
    embedding_config["grid"]["vocab_size"] = len(traj_data.grid)
    embedding_config["roadnet"]["vocab_size"] = traj_data.roadnet.edge_num

    return traj_data, grid_graph_data, road_graph_data

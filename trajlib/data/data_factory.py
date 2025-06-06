from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem

from trajlib.data.data_reader import read_data_chengdu, read_data_geolife, read_data_bj
from trajlib.data.data import GPSTrajData, GridTrajData, GridGraphData, RoadnetTrajData, RoadnetGraphData
from trajlib.data.roadnet_system import RoadnetSystem


def get_bounds(coord_trajs):
    all_lons = [p[0] for t in coord_trajs for p in t]
    all_lats = [p[1] for t in coord_trajs for p in t]
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    return min_lon, min_lat, max_lon, max_lat


def build_grid(coord_trajs, step):
    min_lon, min_lat, max_lon, max_lat = get_bounds(coord_trajs)
    boundary_original = trajdl_cpp.RectangleBoundary(min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat)
    boundary = boundary_original.to_web_mercator()
    grid = SimpleGridSystem(boundary, step_x=step, step_y=step)
    return grid


def build_roadnet(coord_trajs):
    return RoadnetSystem(bounds=get_bounds(coord_trajs), cache_dir="./resource/cache", network_type="all")


def create_data(config):
    data_config = config["data_config"]
    data_path = data_config["data_path"]
    data_size = data_config["data_size"]
    window = data_config["window"]
    varients = data_config["varients"]

    match data_config:
        case {"data_name": "chengdu"}:
            is_roadnet = data_config["data_form"] == "roadnet"
            raw_data = read_data_chengdu(data_path, data_size, clean=is_roadnet, to_gps=is_roadnet)
        case {"data_name": "geolife"}:
            raw_data = read_data_geolife(data_path, data_size)
        case {"data_name": "bj"}:
            raw_data = read_data_bj(data_path, data_size)
        case _:
            raise ValueError()

    match data_config:
        case {"data_form": "gps"}:
            traj_data = GPSTrajData(raw_data, window, varients)
            graph_data = None
        case {"data_form": "grid", "grid_step": step, "unique": unique}:
            grid = build_grid([traj[0] for traj in raw_data], step)
            traj_data = GridTrajData(raw_data, window, varients, grid, unique)
            graph_data = GridGraphData(grid)
            data_config["vocab_size"] = len(grid)
        case {"data_form": "roadnet"}:
            roadnet = build_roadnet([traj[0] for traj in raw_data])
            traj_data = RoadnetTrajData(raw_data, window, varients, roadnet)
            graph_data = RoadnetGraphData(roadnet)
            data_config["vocab_size"] = roadnet.edge_num
        case _:
            raise ValueError()

    return traj_data, graph_data

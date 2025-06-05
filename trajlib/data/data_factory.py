from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem

from trajlib.data.data_reader.chengdu import read_data_chengdu
from trajlib.data.data import GPSTrajData, GridTrajData, GridGraphData, RoadnetTrajData, RoadnetGraphData
from trajlib.data.roadnet_system import RoadnetSystem


def get_bounds(coord_trajs):
    all_lons = [p[0] for t in coord_trajs for p in t]
    all_lats = [p[1] for t in coord_trajs for p in t]
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    return min_lon, min_lat, max_lon, max_lat


def build_grid(coord_trajs, step=100):
    min_lon, min_lat, max_lon, max_lat = get_bounds(coord_trajs)
    boundary_original = trajdl_cpp.RectangleBoundary(min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat)
    boundary = boundary_original.to_web_mercator()
    grid = SimpleGridSystem(boundary, step_x=step, step_y=step)
    return grid


def build_roadnet(coord_trajs):
    return RoadnetSystem(bounds=get_bounds(coord_trajs), cache_dir="./resource/cache", network_type="all")


def create_data(config):
    data_config = config["data_config"]

    match data_config:
        case {"data_name": "chengdu", "data_path": path, "data_size": size, "data_form": form}:
            is_roadnet = form == "roadnet"
            raw_data = read_data_chengdu(path, size, clean=is_roadnet, to_gps=is_roadnet)
        case _:
            return ValueError()

    match data_config:
        case {"data_form": "gps"}:
            traj_data = GPSTrajData(raw_data)
            graph_data = None
        case {"data_form": "grid"}:
            grid = build_grid([traj[0] for traj in raw_data])
            traj_data = GridTrajData(raw_data, grid)
            graph_data = GridGraphData(grid)
            data_config["vocab_size"] = len(grid)
        case {"data_form": "roadnet"}:
            roadnet = build_roadnet([traj[0] for traj in raw_data])
            traj_data = RoadnetTrajData(raw_data, roadnet)
            graph_data = RoadnetGraphData(roadnet)
            data_config["vocab_size"] = roadnet.edge_num
        case _:
            raise ValueError()

    return traj_data, graph_data

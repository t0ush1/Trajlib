from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem

from trajlib.data.data_reader.chengdu import read_data_chengdu
from trajlib.data.data import GPSTrajData, GridTrajData, GridGraphData


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

    match data_config:
        case {"data_name": "chengdu", "data_path": path, "data_size": size}:
            raw_data = read_data_chengdu(path, size)
        case _:
            return ValueError()

    match data_config:
        case {"data_form": "gps"}:
            traj_data = GPSTrajData(raw_data)
            graph_data = None
        case {"data_form": "grid", "grid_step": step}:
            grid = build_grid([traj[0] for traj in raw_data], step=step)
            traj_data = GridTrajData(raw_data, grid)
            graph_data = GridGraphData(grid)
            data_config["vocab_size"] = len(grid)
        case _:
            raise ValueError()

    return traj_data, graph_data

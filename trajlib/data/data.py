from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem

from trajlib.data.trajectory import Trajectory, GPSTrajectory, GridTrajectory


class TrajData:
    trajectories: list[Trajectory] = None


class GraphData:
    def get_nodes(self):
        raise NotImplementedError()

    def get_neighbors(self, node):
        raise NotImplementedError()


class GPSData(TrajData):
    def __init__(self, trajectories):
        self.trajectories = [GPSTrajectory(i, *traj) for i, traj in enumerate(trajectories)]


class GridData(TrajData, GraphData):
    def __init__(self, trajectories, step=100):
        self.coord_trajs = [traj[0] for traj in trajectories]
        self.grid = self._build_grid(step)
        self.trajectories = [GridTrajectory(i, *traj, self.grid) for i, traj in enumerate(trajectories)]

    def _build_grid(self, step):
        all_lons = [p[0] for t in self.coord_trajs for p in t]
        all_lats = [p[1] for t in self.coord_trajs for p in t]
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)

        boundary_original = trajdl_cpp.RectangleBoundary(min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat)
        boundary = boundary_original.to_web_mercator()

        grid = SimpleGridSystem(boundary, step_x=step, step_y=step)
        return grid

    def get_nodes(self):
        return list(range(len(self.grid)))

    def get_neighbors(self, node):
        neighbors = []
        x, y = self.grid.to_grid_coordinate(str(node))
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid.num_x_grids and 0 <= ny < self.grid.num_y_grids:
                nloc = self.grid.locate_by_grid_coordinate(nx, ny)
                neighbors.append(int(nloc))
        return neighbors

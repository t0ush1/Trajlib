from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem

from trajlib.data.trajectory import Trajectory, GPSTrajectory, GridTrajectory


class TrajData:
    def __init__(self, trajectories):
        self.trajectories: list[Trajectory] = trajectories


class GPSTrajData(TrajData):
    def __init__(self, trajectories):
        super().__init__([GPSTrajectory(i, *traj) for i, traj in enumerate(trajectories)])


class GridTrajData(TrajData):
    def __init__(self, trajectories, grid: SimpleGridSystem):
        super().__init__([GridTrajectory(i, *traj, grid) for i, traj in enumerate(trajectories)])
        self.grid = grid


class GraphData:
    def get_nodes(self):
        raise NotImplementedError()

    def get_neighbors(self, node):
        raise NotImplementedError()


class GridGraphData(GraphData):
    def __init__(self, grid: SimpleGridSystem):
        self.grid = grid

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

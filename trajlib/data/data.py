from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem
import torch
from torch_geometric.data import Data as GeoData

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
    def __init__(self, nodes, neighbors, features):
        self.nodes: list[int] = nodes
        self.neighbors: list[list[int]] = neighbors
        self.features: list[list[float]] = features

    def to_geo_data(self, index_as_features=True) -> GeoData:
        if index_as_features:
            x = torch.tensor(self.nodes)
        else:
            x = torch.tensor(self.features).float()
        edge_index = []
        for node in self.nodes:
            for neighbor in self.neighbors[node]:
                edge_index.append([node, neighbor])
        edge_index = torch.tensor(edge_index).t()
        return GeoData(x=x, edge_index=edge_index)


class GridGraphData(GraphData):
    def __init__(self, grid: SimpleGridSystem):
        self.grid = grid
        nodes = list(range(len(self.grid)))
        super().__init__(
            nodes=nodes,
            neighbors=[self.__get_neighbors__(node) for node in nodes],
            features=[self.__get_features__(node) for node in nodes],
        )

    def __get_neighbors__(self, node):
        neighbors = []
        x, y = self.grid.to_grid_coordinate(str(node))
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid.num_x_grids and 0 <= ny < self.grid.num_y_grids:
                nloc = self.grid.locate_by_grid_coordinate(nx, ny)
                neighbors.append(int(nloc))
        return neighbors

    def __get_features__(self, node):
        x, y = self.grid.to_grid_coordinate(str(node))
        return self.grid.get_centroid_of_grid(x, y)

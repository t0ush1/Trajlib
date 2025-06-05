import random
from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem
import torch
from torch_geometric.data import Data as GeoData

from trajlib.data.roadnet_system import RoadnetSystem


SPECIAL_TOKENS = {
    "pad": -1,
    "mask": -2,
    "ignore": -100,
}


class Trajectory:
    def __init__(self, traj_id, locations, timestamps, attributes=None):
        self.traj_id: int = traj_id
        self.locations: list[int | list[float]] = locations
        self.timestamps: list[int] = timestamps
        self.attributes: dict = attributes

    def __len__(self):
        return len(self.locations)


class TrajData:
    def __init__(self, trajectories):
        self.original: list[Trajectory] = []
        self.cropped: list[Trajectory] = []
        self.distorted: list[Trajectory] = []
        for i, traj in enumerate(trajectories):
            self.original.append(self.__transform__(i, *traj))
            self.cropped.append(self.__transform__(i, *self.__crop__(*traj)))
            self.distorted.append(self.__transform__(i, *self.__distort__(*traj)))

    def __len__(self):
        return len(self.original)

    def __transform__(self, traj_id, coordinates, timestamps):
        raise NotImplementedError()

    def __crop__(self, coordinates, timestamps, ratio=0.8):
        n = len(coordinates)
        if n <= 2:
            return coordinates, timestamps
        mid_indices = list(range(1, n - 1))
        keep_count = int(ratio * len(mid_indices))
        keep_mid = sorted(random.sample(mid_indices, keep_count))
        keep_indices = [0] + keep_mid + [n - 1]
        coordinates = [coordinates[i] for i in keep_indices]
        timestamps = [timestamps[i] for i in keep_indices]
        return coordinates, timestamps

    def __distort__(self, coordinates, timestamps):
        coords = []
        for lon, lat in coordinates:
            point = trajdl_cpp.convert_gps_to_webmercator(lon, lat)
            x = point.x + 30 * random.gauss(0, 1)
            y = point.y + 30 * random.gauss(0, 1)
            point = trajdl_cpp.convert_webmercator_to_gps(x, y)
            coords.append([point.lng, point.lat])
        return coords, timestamps


class GPSTrajData(TrajData):
    def __init__(self, trajectories):
        super().__init__(trajectories)

    def __transform__(self, traj_id, coordinates, timestamps):
        return Trajectory(traj_id, coordinates, timestamps)


class GridTrajData(TrajData):
    def __init__(self, trajectories, grid: SimpleGridSystem):
        self.grid = grid
        super().__init__(trajectories)

    def __transform__(self, traj_id, coordinates, timestamps):
        self.grid_locations = []
        self.grid_coordinates = []
        for lon, lat in coordinates:
            point = trajdl_cpp.convert_gps_to_webmercator(lon, lat)
            loc = self.grid.locate_unsafe(point.x, point.y)
            x, y = self.grid.to_grid_coordinate_unsafe(loc)
            if self.grid.in_boundary_by_grid_coordinate(x, y):
                self.grid_locations.append(int(loc))
                self.grid_coordinates.append([x, y])
        return Trajectory(traj_id, self.grid_locations, timestamps)


class RoadnetTrajData(TrajData):
    def __init__(self, trajectories, road_net: RoadnetSystem):
        self.road_net = road_net
        super().__init__(trajectories)

    def __transform__(self, traj_id, coordinates, timestamps):
        osmids, _, timestamps = self.road_net.get_road_osmids_for_points(coordinates=coordinates, timestamps=timestamps)
        return Trajectory(traj_id, osmids, timestamps)


class GraphData:
    def __init__(self, nodes, neighbors, features):
        self.nodes: list[int] = nodes
        self.neighbors: list[list[int]] = neighbors
        self.features: list[list[float]] = features

    def to_geo_data(self, index_as_features=True) -> GeoData:
        x = torch.tensor(self.nodes) if index_as_features else torch.tensor(self.features).float()
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


class RoadnetGraphData(GraphData):
    def __init__(self, road_net: RoadnetSystem):
        self.road_net = road_net

    def get_nodes(self):
        """
        获取所有路段节点（返回osmid列表）
        :return: 所有路段osmid的列表
        """
        return list(range(self.road_net.edge_num))

    def get_neighbors(self, node):
        """
        获取指定路段的完整相邻路段（考虑双向连接）
        :param node: 路段osmid
        :return: 相邻路段osmid列表
        """
        neighbors = []
        connected_vecs = []
        # 筛选目标路段（处理可能的数据类型或列表形式的osmid）
        # 假设node是你要查找的值
        matched_rows = self.road_net.edges[self.road_net.edges["osmid"] == node]

        if not matched_rows.empty:
            for i in range(len(matched_rows)):
                u_node, v_node, _ = matched_rows.index[i]
                if u_node not in connected_vecs:
                    connected_vecs.append(u_node)
                if v_node not in connected_vecs:
                    connected_vecs.append(v_node)
        else:
            return []

        for vec in connected_vecs:
            # u和v应该是等价的
            neighbor = self.road_net.edges[self.road_net.edges.index.get_level_values("u") == vec]
            neighbor = neighbor["osmid"].values.tolist()
            for i in range(len(neighbor)):
                if neighbor[i] not in neighbors:
                    neighbors.append(neighbor[i])

        # 去重并移除自身
        if node in neighbors:
            neighbors.remove(node)
        return neighbors

    def get_features(self, node):
        pass  # TODO

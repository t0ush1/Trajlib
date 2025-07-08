from enum import Enum
import random
from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem
import torch
from torch_geometric.data import Data as GeoData
from tqdm import tqdm

from trajlib.data.roadnet_system import RoadnetSystem


class SpecialToken(float, Enum):
    PAD = -1
    MASK = -2
    UNK = -100


class Trajectory:
    def __init__(self, traj_id, coordinates, grid_ids, road_ids, timestamps, attributes):
        self.traj_id: int = traj_id
        self.coordinates: list[tuple[float, float]] = coordinates
        self.grid_ids: list[int] = grid_ids
        self.road_ids: list[int] = road_ids
        self.timestamps: list[int] = timestamps
        self.attributes: dict = attributes

    def to_tensor(self):
        coords_tensor = torch.tensor(self.coordinates, dtype=torch.float32)
        grids_tensor = torch.tensor(self.grid_ids, dtype=torch.int64)
        roads_tensor = torch.tensor(self.road_ids, dtype=torch.int64)
        times_tensor = torch.tensor(self.timestamps, dtype=torch.int64)
        return coords_tensor, grids_tensor, roads_tensor, times_tensor

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        if isinstance(index, int):
            return (
                self.coordinates[index],
                self.grid_ids[index],
                self.road_ids[index],
                self.timestamps[index],
            )
        elif isinstance(index, slice):
            return Trajectory(
                self.traj_id,
                self.coordinates[index],
                self.grid_ids[index],
                self.road_ids[index],
                self.timestamps[index],
                self.attributes,
            )
        elif isinstance(index, list):
            return Trajectory(
                self.traj_id,
                [self.coordinates[i] for i in index],
                [self.grid_ids[i] for i in index],
                [self.road_ids[i] for i in index],
                [self.timestamps[i] for i in index],
                self.attributes,
            )


class TrajData:
    def __init__(self, trajectories, window, grid, roadnet):
        self.original: list[Trajectory] = []
        self.cropped: list[Trajectory] = []
        self.distorted: list[Trajectory] = []
        self.variants: dict[str, list[Trajectory]] = {
            "original": self.original,
            "cropped": self.cropped,
            "distorted": self.distorted,
        }

        self.grid: SimpleGridSystem = grid
        self.roadnet: RoadnetSystem = roadnet

        threshold, window_size, stride = window

        all_coords = []
        for coords, _, _ in trajectories:
            all_coords.extend(coords)
        all_coords = torch.tensor(all_coords)
        coords_mean = all_coords.mean(dim=0)
        coords_std = all_coords.std(dim=0)

        def generator(traj_id, coordinates, timestamps, attributes):
            grid_ids = self.__map_grid__(coordinates)
            road_ids = self.__map_roadnet__(coordinates)
            
            self.unk_nums.append(road_ids.count(SpecialToken.UNK))
            
            norm_coords = ((torch.tensor(coordinates) - coords_mean) / coords_std).tolist()
            trajectory = Trajectory(traj_id, norm_coords, grid_ids, road_ids, timestamps, attributes)
            for pos in range(0, max(1, len(trajectory) - window_size), stride):
                if len(trajectory) - pos < threshold:
                    break
                yield trajectory[pos : pos + window_size]

        self.unk_nums = []
        for i, (coordinates, timestamps, attributes) in tqdm(
            enumerate(trajectories), desc=f"Generating trajectories", total=len(trajectories)
        ):
            for traj in generator(i, coordinates, timestamps, attributes):
                self.original.append(traj)
                self.cropped.append(self.__crop__(traj))
            for traj in generator(i, self.__distort__(coordinates), timestamps, attributes):
                self.distorted.append(traj)
        print(f"[TrajData] original traj num with UNK: {len(self.unk_nums)}")
        print("[TrajData] UNK num / tot num:", *self.unk_nums, sep="\n")

    def __len__(self):
        return len(self.original)

    def __crop__(self, traj, ratio=0.8):
        n = len(traj)
        if n <= 2:
            return [0, n - 1]
        mid_indices = list(range(1, n - 1))
        keep_count = int(ratio * len(mid_indices))
        keep_mid = sorted(random.sample(mid_indices, keep_count))
        keep_indices = [0] + keep_mid + [n - 1]
        return traj[keep_indices]

    def __distort__(self, coordinates, radius=30):
        coords = []
        for lon, lat in coordinates:
            point = trajdl_cpp.convert_gps_to_webmercator(lon, lat)
            x = point.x + radius * random.gauss(0, 1)
            y = point.y + radius * random.gauss(0, 1)
            point = trajdl_cpp.convert_webmercator_to_gps(x, y)
            coords.append([point.lng, point.lat])
        return coords

    def __map_grid__(self, coordinates):
        grid_ids = []
        min_x, max_x = self.grid.boundary.min_x, self.grid.boundary.max_x - 1
        min_y, max_y = self.grid.boundary.min_y, self.grid.boundary.max_y - 1
        for lon, lat in coordinates:
            point = trajdl_cpp.convert_gps_to_webmercator(lon, lat)
            x = min(max_x, max(min_x, point.x))
            y = min(max_y, max(min_y, point.y))
            loc = self.grid.locate(x, y)
            grid_ids.append(int(loc))
        return grid_ids

    def __map_roadnet__(self, coordinates):
        road_ids = self.roadnet.get_road_osmids_for_points(coordinates)[0]
        road_ids += [SpecialToken.UNK] * (len(coordinates) - len(road_ids))
        return road_ids


class GraphData:
    def __init__(self, nodes, neighbors, features):
        self.nodes: list[int] = nodes
        self.neighbors: list[list[int]] = neighbors
        self.features: list[list[float]] = features

    def __len__(self):
        return len(self.nodes)

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
        nodes = list(range(len(grid)))
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
    def __init__(self, roadnet: RoadnetSystem):
        self.roadnet = roadnet
        nodes = list(range(roadnet.edge_num))
        super().__init__(
            nodes=nodes,
            neighbors=[self.__get_neighbors__(node) for node in nodes],
            features=[self.__get_features__(node) for node in nodes],
        )

    def __get_neighbors__(self, node):
        """
        获取指定路段的完整相邻路段（考虑双向连接）
        :param node: 路段osmid
        :return: 相邻路段osmid列表
        """
        neighbors = []
        connected_vecs = []
        # 筛选目标路段（处理可能的数据类型或列表形式的osmid）
        # 假设node是你要查找的值
        matched_rows = self.roadnet.edges[self.roadnet.edges["osmid"] == node]

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
            neighbor = self.roadnet.edges[self.roadnet.edges.index.get_level_values("u") == vec]
            neighbor = neighbor["osmid"].values.tolist()
            for i in range(len(neighbor)):
                if neighbor[i] not in neighbors:
                    neighbors.append(neighbor[i])

        # 去重并移除自身
        if node in neighbors:
            neighbors.remove(node)
        return neighbors

    def __get_features__(self, node):
        pass  # TODO 返回道路中心的gps坐标

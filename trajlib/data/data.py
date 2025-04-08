from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem
<<<<<<< HEAD
import bisect
from trajlib.data.trajectory import Trajectory, GPSTrajectory, GridTrajectory, RoadNetTrajectory
from trajlib.data.RoadNetSystem import RoadNetSystem
=======
import torch
from torch_geometric.data import Data as GeoData

from trajlib.data.trajectory import Trajectory, GPSTrajectory, GridTrajectory

>>>>>>> d46ad5d9cc834b3e78370095a479189c6a674282

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


class RoadNetTrajData(TrajData):
    def __init__(self, trajectories, road_net: RoadNetSystem):
        super().__init__([RoadNetTrajectory(i, *traj, road_net) for i, traj in enumerate(trajectories)])
        self.road_net = road_net


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

<<<<<<< HEAD
class RoadNetGraphData(GraphData):
    def __init__(self, road_net: RoadNetSystem):
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
        matched_rows = self.road_net.edges[self.road_net.edges['osmid'] == node]

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
            neighbor = self.road_net.edges[self.road_net.edges.index.get_level_values('u') == vec]
            neighbor = neighbor['osmid'].values.tolist()
            for i in range(len(neighbor)):
                if neighbor[i] not in neighbors:
                    neighbors.append(neighbor[i])

        # 去重并移除自身
        if node in neighbors:
            neighbors.remove(node)
        return neighbors
=======
    def __get_features__(self, node):
        x, y = self.grid.to_grid_coordinate(str(node))
        return self.grid.get_centroid_of_grid(x, y)
>>>>>>> d46ad5d9cc834b3e78370095a479189c6a674282

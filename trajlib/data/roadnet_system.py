import os
import osmnx as ox
import geopandas as gpd
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
from typing import List, Tuple


class RoadnetSystem:
    """道路网络元数据管理类，用于处理道路网络数据的获取和轨迹匹配"""

    def __init__(self, bounds: List[float], cache_dir: str, network_type: str = "drive"):
        """
        初始化道路网络元数据

        :param bounds: 地理边界坐标 [north, south, east, west]
        :param cache_dir: 缓存目录路径
        :param network_type: 道路网络类型，默认为"drive"
        """
        self.bounds = bounds
        self.cache_dir = cache_dir
        self.network_type = network_type
        self.map_con, self.nodes, self.edges, self.edge_num = self._get_roadnetwork()
        self.edges = self.edges.sort_index()  # 若已是 MultiIndex，直接排序

        # # 添加道路中心点坐标
        # self.edges['lon'] = self.edges.centroid.x
        # self.edges['lat'] = self.edges.centroid.y

    def _get_roadnetwork(self) -> Tuple[InMemMap, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """获取道路网络数据并构建内存地图"""
        bounds_name = "_".join([str(num).replace(".", "-") for num in self.bounds])
        cache_filepath = os.path.join(self.cache_dir, f"cache_{self.network_type}_{bounds_name}.graphml")

        # 从缓存加载或下载新数据
        if os.path.exists(cache_filepath):
            G = ox.load_graphml(cache_filepath)
        else:
            G = ox.graph_from_bbox(self.bounds, network_type=self.network_type)
            ox.save_graphml(G, cache_filepath)

        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

        # 重新分配edges的osmid，确保相同原始osmid对应相同新id
        edges = edges.reset_index()  # 将MultiIndex转换为列
        # 处理osmid为可哈希类型（例如将列表转换为元组）
        edges["_osmid_hashable"] = edges["osmid"].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        # 创建 {原始osmid: 新id} 的映射字典
        unique_osmid = edges["_osmid_hashable"].unique()
        osmid_to_new_id = {osmid: idx for idx, osmid in enumerate(unique_osmid)}
        # 替换osmid列并清理临时列
        edges["osmid"] = edges["_osmid_hashable"].map(osmid_to_new_id)
        edges = edges.drop(columns=["_osmid_hashable"])
        # 恢复原始索引结构（可选）
        edges = edges.set_index(["u", "v", "key"])

        # 构建内存地图
        map_con = InMemMap(name="pNEUMA", use_latlon=True, use_rtree=True, index_edges=True)

        # 添加节点和边到地图
        for node_id, row in nodes.iterrows():
            map_con.add_node(node_id, (row["y"], row["x"]))
        for node_id_1, node_id_2, _ in G.edges:
            map_con.add_edge(node_id_1, node_id_2)

        return map_con, nodes, edges, len(unique_osmid)

    def get_road_osmids_for_points(
        self, coordinates: List[Tuple[float, float]], timestamps, filter_method="cut"
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        根据轨迹点坐标获取对应的路段osmid和原始(u,v)节点对

        :param coordinates: 经纬度坐标列表 [(lon1, lat1), (lon2, lat2), ...]
        :return: (osmids列表, 原始(u,v)节点对列表)
                如果某点未匹配到路段，对应位置为None
        """
        # 执行GPS路径匹配
        path = [(lon, lat) for (lon, lat) in coordinates]
        # print("len_path:", len(path))
        states_list, matcher = self._match_gps_path(path)
        # print("len_states:", len(states_list))
        # print(states_list)

        osmids = []
        uv_pairs = []

        for u, v in states_list:
            uv_pairs.append((u, v))
            osmid = self._get_osmid_for_edge(u, v)
            osmids.append(osmid)

        # print(len(timestamps), len(osmids))
        if len(timestamps) != len(osmids):
            mmviz.plot_map(
                self.map_con,
                matcher=matcher,
                show_labels=False,
                show_matching=True,
                filename=None,
                figwidth=5,
            )
            if filter_method == "delete":
                # TODO
                osmids = []
                timestamps = []
            elif filter_method == "cut":
                timestamps = timestamps[0 : len(osmids)]
                if len(osmids) == 0:
                    print("error:no osmids!")
        return osmids, uv_pairs, timestamps

    def _match_gps_path(
        self, lon_lat_list: List[Tuple[float, float]], visualization: bool = False
    ) -> Tuple[List[Tuple[int, int]], DistanceMatcher]:
        """单个GPS路径匹配"""
        path = [(lat, lon) for (lon, lat) in lon_lat_list]  # 转换坐标顺序

        matcher = DistanceMatcher(
            self.map_con,
            max_dist=300,
            max_dist_init=250,
            min_prob_norm=0.0001,
            non_emitting_length_factor=0.95,
            obs_noise=50,
            obs_noise_ne=50,
            dist_noise=50,
            max_lattice_width=20,
            non_emitting_states=False,
        )
        # matcher = DistanceMatcher(
        #     self.map_con,
        #     # 关键参数调整
        #     max_dist=1000,          # 从300增加到1000米（覆盖更大定位偏差）
        #     max_dist_init=500,      # 从250增加到500米（放宽初始搜索范围）
        #     min_prob_norm=0.01,     # 从0.0001调高到0.01（保留更多低概率候选）
        #     non_emitting_length_factor=0.98,  # 从0.95调高到0.98（降低长路径惩罚）

        #     # 噪声参数调整
        #     obs_noise=100,          # 从50增加到100米（适配更大GPS误差）
        #     obs_noise_ne=150,       # 非发射状态使用更高噪声容忍度
        #     dist_noise=100,         # 与obs_noise同步调整

        #     # 结构参数优化
        #     max_lattice_width=50,  # 从20增加到100（保留更多候选路径）
        #     non_emitting_states=True,

        #     # 新增调试参数
        #     avoid_goingback=False   # 暂时关闭回退惩罚（测试是否因转向惩罚过严导致）
        # )

        states, _ = matcher.match(path, unique=False)

        if visualization:
            mmviz.plot_map(
                self.map_con,
                matcher=matcher,
                show_labels=False,
                show_matching=True,
                filename=None,
                figwidth=5,
            )
        return states, matcher

    def _get_osmid_for_edge(self, u: int, v: int) -> str:
        """根据节点u,v获取路段的osmid"""
        try:
            # 直接通过多级索引 (u, v) 定位行
            # 假设一个索引定位到唯一的一行，否则：return self.edges.loc[(u, v), "osmid"].iloc[0]
            return self.edges.loc[(u, v, 0), "osmid"]
        except KeyError:
            # 若索引不存在，返回 None
            print("无法定位路段id")
            return None

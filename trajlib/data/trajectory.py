import numpy as np

from trajdl.grid import SimpleGridSystem
from trajdl import trajdl_cpp
from trajlib.data.RoadNetSystem import RoadNetSystem

class Trajectory:
    def __init__(self, traj_id, locations, timestamps):
        self.traj_id: int = traj_id
        self.locations: list[int | list[float]] = locations
        self.timestamps: list[int] = timestamps

    def __len__(self):
        return len(self.locations)


class GPSTrajectory(Trajectory):
    def __init__(self, traj_id, coordinates, timestamps):
        super().__init__(traj_id, coordinates, timestamps)


class GridTrajectory(Trajectory):
    def __init__(self, traj_id, coordinates, timestamps, grid: SimpleGridSystem):
        self.grid_locations = []
        self.grid_coordinates = []
        for lon, lat in coordinates:
            point = trajdl_cpp.convert_gps_to_webmercator(lon, lat)
            loc = grid.locate_unsafe(point.x, point.y)
            self.grid_locations.append(int(loc))
            self.grid_coordinates.append(grid.to_grid_coordinate(loc))
        super().__init__(traj_id, self.grid_locations, timestamps)

class RoadNetTrajectory(Trajectory):
    def __init__(self, traj_id, coordinates, timestamps, road_net: RoadNetSystem):
        osmids, _, timestamps = road_net.get_road_osmids_for_points(coordinates=coordinates, timestamps=timestamps)
        # print("RoadNetTrajectory:", len(osmids), len(timestamps))
        super().__init__(traj_id, osmids, timestamps)
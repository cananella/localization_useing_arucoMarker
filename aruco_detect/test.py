import matplotlib.axes
from socket_video_server import VideoStreamServer
import aruco_marker_detect
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import threading
from collections import deque

class ArucoLocalization:
    def __init__(self, host, port, calibration_file, grid_size=(450, 450), grid_map_file=None):
        self.video_server = VideoStreamServer(host=host, port=port)
        self.grid_size = grid_size
        self.world = np.full(grid_size, 1 / (grid_size[0] * grid_size[1]))
        self.measurements = []
        self.before_measurements = []

        self.dq = deque([[] for _ in range(11)], maxlen=11)
        
        self.unupdate_cnt = 0
        self.camera_thread_on = True
        self.grid_map_file = grid_map_file
        self.grid_map = cv2.imread(grid_map_file) if grid_map_file else None
        if self.grid_map is not None:
            self.grid_map = cv2.cvtColor(self.grid_map, cv2.COLOR_BGR2RGB)
            self.grid_map = cv2.resize(self.grid_map, (grid_size[1], grid_size[0]))

        self.marker_positions = [(300, 100), (0, 50), (50, 0), (150, 25), (50, 200), None, (400, 200), (250,25), None, (450, 50), (75,25), (350, 25), None]

        self.fig, self.ax = plt.subplots()
        self.ani = FuncAnimation(self.fig, self.animate, blit=True)

        self.detect_thread = threading.Thread(target=self.aruco_detect_thread, daemon=True)
        if self.camera_thread_on:
            self.camera_matrix, self.dist_coeffs = aruco_marker_detect.load_camera_parameters(calibration_file)
            self.detect_thread.start()

    def init_world(self):
        self.world = np.full(self.grid_size, 1 / (self.grid_size[0] * self.grid_size[1]))

    def aruco_detect_thread(self):
        self.video_server.accept_connection()
        self.video_server.cam_ui = True
        while True:
            self.video_server.getframe()
            frame, detectdata = aruco_marker_detect.detect_markers(self.video_server.frame, self.camera_matrix, self.dist_coeffs, 0.053)
            self.measurements = [(id, round(distance * 100)) for id, _, _, distance in detectdata]
            cv2.imshow('IMG', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def update_belief(self, world, marker_positions, measurements, noise=10):
        if not measurements:
            return world
        new_world = np.zeros_like(world)
        X, Y = np.meshgrid(np.arange(self.grid_size[1]), np.arange(self.grid_size[0]))

        for idx, z in measurements:
            if marker_positions[idx[0]] is not None:
                marker_pos = np.array(marker_positions[idx[0]])
                distances = np.sqrt((X - marker_pos[0]) ** 2 + (Y - marker_pos[1]) ** 2)
                prob = np.exp(-((distances - z) ** 2) / (2 * noise ** 2))
                new_world += prob

        new_world *= world
        return new_world / new_world.sum()

    def plot_world(self, ax, world, marker_positions):
        ax.clear()
        if self.grid_map is not None:
            ax.imshow(self.grid_map, extent=[0, self.grid_size[1], self.grid_size[0], 0])

        im = ax.imshow(world, cmap='Blues', alpha=0.5)
        scatter_markers = []
        for pos in marker_positions:
            if pos is None:
                continue
            scatter_markers.append(ax.scatter([pos[0]], [pos[1]], color='red', label='Markers'))
        
        max_prob_position = np.unravel_index(np.argmax(world), world.shape)
        scatter_max_prob = ax.scatter([max_prob_position[1]], [max_prob_position[0]], color='yellow', edgecolors='black', label='Highest Probability')
        print("prob position = ", max_prob_position)
        
        for idx, z in self.measurements:
            marker_pos = marker_positions[idx[0]]
            if marker_pos is not None:
                ax.plot([marker_pos[0], max_prob_position[1]], [marker_pos[1], max_prob_position[0]], color='green', linestyle='--')
        
        ax.set_title("Histogram Filter Localization")
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.invert_yaxis()
        return [im, scatter_max_prob] + scatter_markers

    def animate(self, frame):
        if self.measurements:
            same_pos_flag = (len(self.measurements) == len(self.before_measurements)) and all(
                any(before_id == id and abs(z - before_z) < 5 for before_id, before_z in self.before_measurements)
                for id, z in self.measurements
            )

            if not same_pos_flag:
                self.dq.append(self.measurements)
                self.before_measurements = self.measurements.copy()
                print('world update')
                self.init_world()
                for elem in self.dq:
                    self.world = self.update_belief(self.world, self.marker_positions, elem)

        return self.plot_world(self.ax, self.world, self.marker_positions)

    def show(self):
        plt.show()

# Example usage:
if __name__ == "__main__":
    localization = ArucoLocalization(host='', port=8888, calibration_file='calibration_params.yaml', grid_map_file='map.png')
    localization.show()

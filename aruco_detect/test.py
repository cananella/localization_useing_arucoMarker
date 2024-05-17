import matplotlib.axes
from socket_video_server import VideoStreamServer
import aruco_marker_detect
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import threading

def aruco_detec_thread():
    global measurements
    video_server.accept_connection()
    video_server.cam_ui = True
    camera_matrix, dist_coeffs = aruco_marker_detect.load_camera_parameters('agv_localization/calibration_params.yaml')
    while True:
        video_server.getframe()
        detectdata = []
        measurements = []
        frame, detectdata = aruco_marker_detect.detect_markers(video_server.frame, camera_matrix, dist_coeffs, 0.053)
        for id, marker_pos, rot, distance in detectdata:
            measurements.append((id, distance*100))
        cv2.imshow('IMG', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def update_belief(world, marker_positions, measurements, noise=10):
    """Belief 업데이트 함수"""
    if not measurements:
        return world
    new_world = np.zeros_like(world)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            prob = 1.0
            for idx, z in measurements:

                distance = np.sqrt((i - marker_positions[idx[0]][0])**2 + (j - marker_positions[idx[0]][1])**2)
                prob *= np.exp(-((distance - z)**2) / (2 * noise**2))
            new_world[i, j] = prob
    new_world *= world
    return new_world / new_world.sum()

def plot_world(ax, world, marker_positions):
    """월드 상태 플로팅"""
    ax.clear()
    im = ax.imshow(world, cmap='Blues')
    scatter_markers = [ax.scatter([pos[1]], [pos[0]], color='red', label='Markers') for pos in marker_positions]  # 마커 위치
    # 최대 확률 위치 찾기 및 표시
    max_prob_position = np.unravel_index(np.argmax(world), world.shape)
    scatter_max_prob = ax.scatter([max_prob_position[1]], [max_prob_position[0]], color='yellow', edgecolors='black', label='Highest Probability')
    ax.set_title("Histogram Filter Localization")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.legend()
    return [im, scatter_max_prob] + scatter_markers

def animate(frame):
    global world

    print('measurements: ', measurements)
    if measurements:
        print('world update')
        world = update_belief(world, marker_positions, measurements)

    return plot_world(ax, world, marker_positions)



video_server = VideoStreamServer(host='', port=8888)
measurements = []
# 마커의 위치 설정 (여러 개의 마커)
marker_positions = [(0, 0), (50, 0), (0, 0), (0, 0), (0, 0), (0, 0), (50, 50), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

detect_thread = threading.Thread(target=aruco_detec_thread, daemon=True)
detect_thread.start()

# 환경 설정
grid_size = (500, 500)  # 격자의 크기
world = np.zeros(grid_size)  # 확률 분포를 위한 격자 초기화

# 초기 위치 확률 분포 설정
world += 1
world /= world.sum()

fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, blit=True)

plt.show()

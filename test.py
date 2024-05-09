from socket_video_server import VideoStreamServer
import aruco_marker_detect
import cv2

VS=VideoStreamServer()
VS.accept_connection() 
camera_matrix, dist_coeffs = aruco_marker_detect.load_camera_parameters('calibration_params.yaml')
while True:
    frame=VS.receive_frame()
    frame = aruco_marker_detect.detect_markers(frame, camera_matrix, dist_coeffs, 0.027)
    cv2.imshow('Detected Markers', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

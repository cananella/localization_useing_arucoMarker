import cv2
import numpy as np
import yaml
import argparse

def detect_markers(image, camera_matrix, dist_coeffs, marker_size):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        if rvecs is not None and tvecs is not None:
            for rvec, tvec, marker_id in zip(rvecs, tvecs, ids):
                rvect, _ = cv2.Rodrigues(rvec)
                marker_pos = np.dot(-rvect.T, tvec)
                marker_pos = marker_pos.flatten()
                print("Marker ID:", marker_id, "Marker position:", marker_pos)
                print("rvec: ",rvec,"   tvec: ",tvec)
            print("")
                
    else:
        print("No markers detected")
    return image

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c,mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


def load_camera_parameters(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
        camera_matrix = np.array(data["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)
        dist_coeffs = np.array(data["distortion_coefficients"]["data"], dtype=np.float32)
    return camera_matrix, dist_coeffs


def main(marker_size=0.105, frame=np.array([0])):
    camera_matrix, dist_coeffs = load_camera_parameters('calibration_params.yaml')
    frame = detect_markers(frame, camera_matrix, dist_coeffs, marker_size)
    cv2.imshow('Detected Markers', frame)
    cv2.waitKey(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect ArUco markers.')
    parser.add_argument('--marker_size', type=float, default=0.105,
                        help='Size of the ArUco markers in meters.')
    args = parser.parse_args()
    main(args.marker_size)
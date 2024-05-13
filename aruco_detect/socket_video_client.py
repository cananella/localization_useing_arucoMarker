import cv2
import socket
import numpy as np

class VideoStreamClient:
    def __init__(self, server_ip, server_port, camera_id=0, width=640, height=480, quality=30):
        self.server_ip = server_ip
        self.server_port = server_port
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.quality = quality
        self.socket = None
        self.cam = None

        self.connect_to_server()
        self.setup_camera()

    def connect_to_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_ip, self.server_port))
        print("Connected to server at {}:{}".format(self.server_ip, self.server_port))

    def setup_camera(self):
        self.cam = cv2.VideoCapture(self.camera_id)
        self.cam.set(3, self.width)  # Set width
        self.cam.set(4, self.height)  # Set height

    def send_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            print("Failed to capture frame")
            return False
        result, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        data = np.array(encoded_frame)
        stringData = data.tobytes()  # Using tobytes instead of tostring (which is deprecated)
        self.socket.sendall((str(len(stringData))).encode().ljust(16) + stringData)
        return True

    def run(self):
        try:
            while True:
                if not self.send_frame():
                    break
        except KeyboardInterrupt:
            print("Interrupted and closing...")
        finally:
            self.close()

            

    def close(self):
        self.cam.release()
        self.socket.close()
        print("Camera and socket closed")

# Usage
if __name__ == "__main__":
    client = VideoStreamClient('127.0.0.1', 8485)
    client.run()
import time
import logging
import signal
import threading
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
import os
import sys

# Get the absolute path of the parent directory (project root) of the current script's directory.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Insert it at the front of the module search path.
sys.path.insert(0, project_root)
from lib.camera_client import UnifiedReceiverClient
import lib.robot_pb2 as pb


GRPC_TARGET = "localhost:50051"

class ImageRecorder:
    def __init__(self, camera_names):
        cli = UnifiedReceiverClient(grpc_target=GRPC_TARGET)
        cli.start()
        
        self.camera_names = camera_names
        self.cli = cli
        self.image_dict = {}
        for cam_name in camera_names:
            threading.Thread(target=self.handler, args=(cam_name,), daemon=True).start()

    def handler(self, cam_name):
        while True:
            item = None
            while item is None:
                item = self.cli.get_latest_frame(cam_name)
                time.sleep(0.01)
            bgr, ts = item
            self.image_dict[cam_name] = bgr

    def get_images(self):
        return  self.image_dict.copy()
            


if __name__ == '__main__':
    # main()
    recorder = ImageRecorder(camera_names=['cam_left_wrist', 'cam_high', 'cam_right_wrist'])
    while True:
        print(recorder.image_dict)
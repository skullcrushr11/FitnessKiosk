# imports
import os
import cv2
import json
import time
import numpy as np
import mediapipe as mp

# local imports
from tools import Tools
from graphics import Graphics

assert os.path.exists("preferences.json"), "[{os.getcwd()}] needs to contain preferences.json file!"

with open("preferences.json", 'r') as f:
    preferences = json.load(f)

class PoseEstimator:
    landmark_indices = preferences["mediapipe"]["pose_landmarks"]
    li = landmark_indices

    def __init__(self):
        self.estimator = mp.solutions.pose.Pose(min_detection_confidence = preferences["mediapipe"]["pose_settings"]["min_detection_confidence"],
                                 min_tracking_confidence = preferences["mediapipe"]["pose_settings"]["min_tracking_confidence"])

    # expected input RGB
    def get_landmarks(self, frame):
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self.estimator.process(frame)
        
        keypoints = []
        pose_landmarks = results.pose_landmarks # NormalizedLandmarkList

        if pose_landmarks == None:
            return None

        land = pose_landmarks.landmark

        for data_point in land:
            keypoints.append({
                'x': data_point.x, 
                'y': data_point.y, 
                'z': data_point.z, 
                'visibility': data_point.visibility,
                'presence': data_point.presence
                })

        return keypoints

    # same as get_joint() but in range [0, 1] (not scaled to image)
    def get_normalised_joint(self, joint, keypoints):
        assert (joint in self.landmark_indices) , f"{joint} is not a valid joint name, see preferences.json"
        assert (len(keypoints) == 33), f"Keypoints argument must contain 33 landmarks!"

        presence = keypoints[self.li[joint]]['presence']
        visibility = keypoints[self.li[joint]]['visibility']

        return (keypoints[self.li[joint]]['x'], keypoints[self.li[joint]]['y'], keypoints[self.li[joint]]['z']), presence, visibility


    # expects string that corresponds to those present in pose_landmarks in preferences.json
    def get_joint(self, joint, keypoints, height, width):
        assert (joint in self.landmark_indices) , f"{joint} is not a valid joint name, see preferences.json"
        assert (len(keypoints) == 33), f"Keypoints argument must contain 33 landmarks!"

        presence = keypoints[self.li[joint]]['presence']
        visibility = keypoints[self.li[joint]]['visibility']

        return (keypoints[self.li[joint]]['x'] * width, keypoints[self.li[joint]]['y'] * height, keypoints[self.li[joint]]['z']), presence, visibility


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    pose = PoseEstimator()

    yellow = (209, 224, 36)
    pink = (234, 0, 255)
    grey = (212, 188, 214)

    while capture.isOpened():
        success, frame = capture.read()
        height, width = frame.shape[:2]

        if not success:
            print("Empty Frame")
            continue

        keypoints = pose.get_landmarks(frame)

        if keypoints != None:
            rw_coords, rw_pres, rw_vis = pose.get_joint("left_wrist", keypoints, height, width)
            rw_coords = rw_coords[:2]

            re_coords, re_pres, re_vis = pose.get_joint("left_elbow", keypoints, height, width)
            re_coords = re_coords[:2]

            rs_coords, rs_pres, rs_vis = pose.get_joint("left_shoulder", keypoints, height, width)
            rs_coords = rs_coords[:2]

            # print(rw_vis, re_vis, rs_vis)

            if rw_vis > 0.7:
                cv2.circle(frame, Tools.int_tuple(rw_coords), 6, grey, -1)

            if re_vis > 0.7:
                cv2.circle(frame, Tools.int_tuple(re_coords), 6, grey, -1)

            if rs_vis > 0.7:
                cv2.circle(frame, Tools.int_tuple(rs_coords), 6, grey, -1)

            if rw_vis > 0.7 and re_vis > 0.7:
                cv2.line(frame, Tools.int_tuple(rw_coords), Tools.int_tuple(re_coords), pink, 3)

            if re_vis > 0.7 and rs_vis > 0.7:
                cv2.line(frame, Tools.int_tuple(re_coords), Tools.int_tuple(rs_coords), pink, 3)

            if all([rw_vis > 0.8, re_vis > 0.8, rs_vis > 0.8]):
                angle = Graphics.get_angle_2d(Graphics.make_euclidean(rw_coords, height, width), Graphics.make_euclidean(re_coords, height, width), Graphics.make_euclidean(rs_coords, height, width))
                cv2.putText(frame, f"Angle: {angle}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        # cv2.imshow("Pose Estimation Test", cv2.flip(frame, 1))
        cv2.imshow("Pose Estimation Test", frame)

        # esc key pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break


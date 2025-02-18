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
from pose_estimation import PoseEstimator

assert os.path.exists("preferences.json"), "[{os.getcwd()}] needs to contain preferences.json file!"

with open("preferences.json", 'r') as f:
    preferences = json.load(f)

class PushupDetector:
    def __init__(self):
        # states of a push up
        self.top_position = False
        self.bottom_position = False
        self.error = False

        # frame measurements
        self.height = None
        self.width = None

        # pose estimator
        self.pose = PoseEstimator()
        self.keypoints = []

        # current locations of joints of interest
        self.right_wrist = None
        self.right_shoulder = None
        self.right_elbow = None
        self.right_hip = None
        self.right_knee = None
        self.right_ankle = None
        self.right_foot = None

        # angles of interest
        self.leg_angle = 0
        self.foot_wrist = 0
        self.upper_body_angle = 0

        # total measurements
        self.pushup_count = 0

        # time that each error has existed
        self.error_times = {
                "foot_wrist": 0,
                "leg_angle": 0,
                "upper_body_angle": 0
                }

    def get_count(self):
        return self.pushup_count

    # new session started
    def reset_push(self):
        self.top_position = False
        self.bottom_position = False

        self.right_wrist = None
        self.right_shoulder = None
        self.right_elbow = None
        self.right_hip = None
        self.right_knee = None
        self.right_ankle = None

        self.pushup_count = 0

    # this method must be manually calling anything else
    def update_joints(self):
        self.right_wrist, _ , self.right_wrist_vis  = self.pose.get_joint("right_wrist", keypoints, self.height, self.width)
        self.right_shoulder, _ , self.right_shoulder_vis= self.pose.get_joint("right_shoulder", keypoints, self.height, self.width)
        self.right_elbow, _ , self.right_elbow_vis = self.pose.get_joint("right_elbow", keypoints, self.height, self.width)
        self.right_hip, _ , self.right_hip_vis = self.pose.get_joint("right_hip", keypoints, self.height, self.width)
        self.right_knee, _ , self.right_knee_vis = self.pose.get_joint("right_knee", keypoints, self.height, self.width)
        self.right_ankle, _ , self.right_ankle_vis = self.pose.get_joint("right_ankle", keypoints, self.height, self.width)
        self.right_foot, _ , self.right_foot_vis = self.pose.get_joint("right_foot", keypoints, self.height, self.width)

        # wrist and foot must be in line
        self.foot_wrist = (abs(self.right_wrist[1] - self.right_foot[1]) / self.height) * 100

        # angle created by foot-knee-hip (must be close to 180)
        self.leg_angle = Graphics.get_angle_2d(Graphics.make_euclidean(self.right_ankle, self.height, self.width),
                                          Graphics.make_euclidean(self.right_knee, self.height, self.width),
                                          Graphics.make_euclidean(self.right_hip, self.height, self.width))

        # angle created by knee-hip-shoulder (must be close to 180)
        self.upper_body_angle = Graphics.get_angle_2d(Graphics.make_euclidean(self.right_knee, self.height, self.width),
                                          Graphics.make_euclidean(self.right_hip, self.height, self.width),
                                          Graphics.make_euclidean(self.right_shoulder, self.height, self.width))

        # angle created by shoulder-elbow-wrist
        self.arm_angle = Graphics.get_angle_2d(Graphics.make_euclidean(self.right_wrist, self.height, self.width),
                                          Graphics.make_euclidean(self.right_elbow, self.height, self.width),
                                          Graphics.make_euclidean(self.right_shoulder, self.height, self.width))

    def check_errors(self):
        current_time = time.time()
        if self.foot_wrist > preferences["pushup"]["error"]["foot_wrist_limit"]:
            if self.error_times["foot_wrist"] == 0: self.error_times["foot_wrist"] = current_time
            if (current_time - self.error_times["foot_wrist"]) > preferences["pushup"]["error"]["error_limit"]:
                print("Foot not in line with wrist!")
                self.error = True
                self.error_times["foot_wrist"] = current_time # reset time
        else:
            self.error_times["foot_wrist"] = current_time

        if self.leg_angle < preferences["pushup"]["error"]["leg_angle_threshold"]:
            if self.error_times["leg_angle"] == 0: self.error_times["leg_angle"] = current_time
            if (current_time - self.error_times["leg_angle"]) > preferences["pushup"]["error"]["error_limit"]:
                print("Leg not straight (~165 deg)!")
                self.error = True
                self.error_times["leg_angle"] = current_time # reset time
        else:
            self.error_times["leg_angle"] = current_time

        if self.upper_body_angle < preferences["pushup"]["error"]["upper_angle_threshold"]:
            if self.error_times["upper_body_angle"] == 0: self.error_times["upper_body_angle"] = current_time
            if (current_time - self.error_times["upper_body_angle"]) > preferences["pushup"]["error"]["error_limit"]:
                print("Upper body not straight (~170 deg)!")
                self.error = True
                self.error_times["upper_body_angle"] = current_time # reset time
        else:
            self.error_times["upper_body_angle"] = current_time

    def check_top_position(self):
        if self.arm_angle > preferences["pushup"]["start"]["angle"]:
            self.top_position = True

    def check_bottom_position(self):
        if self.arm_angle < preferences["pushup"]["end"]["angle"]:
            self.bottom_position = True
            self.top_position = False

    def check_increment(self):
        if self.bottom_position == True and self.top_position == True:
            self.pushup_count += 1
            self.bottom_position = False
            self.top_position = False

    def process_frame(self, height, width, keypoints):
        self.error = False

        self.keypoints = keypoints
        self.height = height
        self.width = width

        self.update_joints()
        self.check_errors()

        if self.error:
            self.reset_push()
            return False

        self.check_top_position()
        self.check_bottom_position()
        self.check_increment()

        return True

    def check_start(self):
        pass

if __name__ == "__main__":
    annotate = True
    using_mp4 = False
    video_path = "data/stock_pushup_footage.mp4"

    if using_mp4:
        capture = cv2.VideoCapture(video_path)
    else:
        capture = cv2.VideoCapture(0)

    pose = PoseEstimator()
    pushup_detector = PushupDetector()

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            if using_mp4:
                break

            print("Empty Frame")
            continue

        height, width = frame.shape[:2]
        keypoints = pose.get_landmarks(frame)

        if keypoints != None:
            if annotate:
                rw = pose.get_joint("right_wrist", keypoints, height, width)
                re = pose.get_joint("right_elbow", keypoints, height, width)
                rs = pose.get_joint("right_shoulder", keypoints, height, width)
                rh = pose.get_joint("right_hip", keypoints, height, width)
                rk = pose.get_joint("right_knee", keypoints, height, width)
                ra = pose.get_joint("right_ankle", keypoints, height, width)

                Graphics.draw_angle(rw, re, rs, keypoints, frame)
                Graphics.draw_angle(ra, rk, rh, keypoints, frame)
                Graphics.draw_angle(rk, rh, rs, keypoints, frame)

            if pushup_detector.process_frame(height, width,  keypoints):
                cv2.putText(frame, f"Pushup Count: {pushup_detector.get_count()}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, Graphics.red, 3)
            else:
                raise Exception("Video stream has error!")

        # cv2.imshow("Pose Estimation Test", cv2.flip(frame, 1))
        cv2.imshow("Pose Estimation Test", frame)

        # esc key pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break


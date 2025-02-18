
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

class squat_detector:
    def __init__(self):
        # states of a squat
        self.top_position = False
        self.bottom_position = False
        self.error = False

        # frame measurements
        self.height = None
        self.width = None

        # pose estimator
        self.pose = PoseEstimator()
        self.keypoints=[]

        # current locations of joints of interest 
        self.right_shoulder=None #12
        self.right_elbow=None #14
        self.right_wrist=None #16
        self.right_hip=None #24
        self.right_knee=None #26
        self.right_ankle=None #28

        #angles of intrest
        self.elbow_angle=0
        self.shoulder_angle=0
        self.knee_angle=0

        #squat_count
        self.squat_count=0

        #number of times error occoured
        self.error_count = {
                "elbow_angle":0,
                "shoulder_angle":0
            }

    def get_count(self):
        return self.squat_count
    

    def reset_squat(self):
        self.top_position = False
        self.bottom_position = False

        self.right_shoulder=None #12
        self.right_elbow=None #14
        self.right_wrist=None #16
        self.right_hip=None #24
        self.right_knee=None #26
        self.right_ankle=None #28 

        self.squat_count=0

    def update_keypoints(self):
        self.right_shoulder, _ , self.right_shoulder_vis= self.pose.get_joint("right_shoulder", keypoints, self.height, self.width)
        self.right_elbow, _ , self.right_elbow_vis = self.pose.get_joint("right_elbow", keypoints, self.height, self.width)
        self.right_wrist, _ , self.right_wrist_vis  = self.pose.get_joint("right_wrist", keypoints, self.height, self.width)
        self.right_hip, _ , self.right_hip_vis = self.pose.get_joint("right_hip", keypoints, self.height, self.width)
        self.right_knee, _ , self.right_knee_vis = self.pose.get_joint("right_knee", keypoints, self.height, self.width)
        self.right_ankle, _ , self.right_ankle_vis = self.pose.get_joint("right_ankle", keypoints, self.height, self.width)   

        #hands should be straight (close to 180 degrees)       
        self.elbow_angle=Graphics.get_angle_2d(Graphics.make_euclidean(self.right_shoulder, self.height, self.width),
                                               Graphics.make_euclidean(self.right_elbow, self.height, self.width),
                                                Graphics.make_euclidean(self.right_shoulder, self.height, self.width))
    
        #hands should be held perpendicular to body (close to 90)
        self.shoulder_angle=Graphics.get_angle_2d(Graphics.make_euclidean(self.right_elbow, self.height, self.width),
                                               Graphics.make_euclidean(self.right_shoulder, self.height, self.width),
                                                Graphics.make_euclidean(self.right_hip, self.height, self.width))

        #angle between knee(keeps changing as per position)
        self.knee_angle=Graphics.get_angle_2d(Graphics.make_euclidean(self.right_hip, self.height, self.width),
                                               Graphics.make_euclidean(self.right_knee, self.height, self.width),
                                                Graphics.make_euclidean(self.right_ankle, self.height, self.width))
        
        #upper body should be straight initially (close to 180)
        self.upper_body_angle = Graphics.get_angle_2d(Graphics.make_euclidean(self.right_knee, self.height, self.width),
                                          Graphics.make_euclidean(self.right_hip, self.height, self.width),
                                          Graphics.make_euclidean(self.right_shoulder, self.height, self.width))
    def get_errors(self):
        current_time= time.time()
        #hands
    '''
    if self.elbow_angle < preferences['squat']['error']['elbow_angle_threshold']:
            if self.error_count["elbow_angle"]==0: self.error_count["elbow_angle"]=current_time
            if (current_time - self.error_count["elbow_angle"]) > preferences["squat"]["error"]["error_limit"]:
                print("Hands not straight")
                self.error=True
                self.error_count["elbow_angle"]= current_time
            else:

        #hands perpendicular to body
        if self.shoulder_angle < preferences["squat"]["error"]["shoulder_angle_threshold"]:
            if self.error_count["shoulder_angle"]==0: self.error_count["shoulder_angle"] =current_time
            if (current_time - self.error_count["shoulder_angle"]) > preferences["squat"]["error"]["error_limit"]:
                print("hands not perpendicular to body (~90 deg)!")
                self.error = True
                self.error_count["shoulder_angle"]=current_time # reset time
        else:
            self.error_count["shoulder_angle"]=current_time
    '''
    def check_top_position(self):
        if self.knee_angle > preferences['squat']['knee_start']['angle'] :#and self.upper_body_angle > preferences['squat']['upper_body_angle']:
            self.top_position=True
    
    def check_bottom_position(self):
        if self.knee_angle < preferences['squat']['knee_end']['angle']:
           
            self.top_position=False
            self.bottom_position=True
    
    def check_increment(self):
        if self.bottom_position==True and self.top_position==True:
            self.squat_count=self.squat_count+1
            self.bottom_position=False
            self.top_position=False

    def process_frame(self,height,width,keypoints):
        self.error=False
        
        self.keypoints = keypoints
        self.height = height
        self.width = width

        self.update_keypoints()
        self.get_errors()

        if self.error:
            self.reset_squat()
            return False
        
        self.check_top_position()
        self.check_bottom_position()
        self.check_increment()

        return True
    
    def check_start(self):
        pass

if __name__ == "__main__": 
    annotate = True
    using_mp4 = True
    video_path = "C:\\Users\\Sai Tarun\\Videos\\EaseUS RecExperts\\20230825_110748.mp4"

    if using_mp4:
        capture = cv2.VideoCapture(video_path)
    else:
        capture = cv2.VideoCapture(0)

    pose = PoseEstimator()
    squatdetector = squat_detector()

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            if using_mp4:
                break
            
            print("Empty Frame")
            continue

        height, width = frame.shape[:2]
        keypoints = pose.get_landmarks(frame)

       # if self.top_position==True
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
            
            if squatdetector.process_frame(height, width, keypoints):
                cv2.putText(frame, f"squat Count: {squatdetector.get_count()}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, Graphics.red, 2)
            else:
                raise Exception("Video stream has error!")

        # cv2.imshow("Pose Estimation Test", cv2.flip(frame, 1))
        cv2.imshow("Pose Estimation Test", frame)

        # esc key pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
        

        



        



                


        




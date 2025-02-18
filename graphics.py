# imports
import cv2
import numpy as np

# local imports
from tools import Tools

class Graphics:
    yellow = (209, 224, 36)
    pink = (234, 0, 255)
    grey = (212, 188, 214)
    green = (88, 230, 11)
    red = (0, 25, 255)

    def __init__(self):
        pass

    # angle a -> b -> c (ie. B)
    @staticmethod
    def get_angle_2d(a, b, c):
        vector_ab = np.array(a) - np.array(b)
        vector_bc = np.array(c) - np.array(b)

        dot_product = np.dot(vector_ab, vector_bc)
        mag_ab = np.linalg.norm(vector_ab)
        mag_bc = np.linalg.norm(vector_bc)

        cos_angle = dot_product / (mag_ab * mag_bc)
        angle_radians = np.arccos(cos_angle)

        cross_product = np.cross(vector_ab, vector_bc)
        angle_degrees = np.degrees(angle_radians)

        # ensure cross produce > 0
        # if np.dot(cross_product, np.array([0, 0, 1])) < 0:
        #     angle_radians = 2 * np.pi - angle_radians

        return angle_degrees

    @staticmethod
    def make_euclidean(a, height, width):
        return (a[0], height - a[1])

    @staticmethod
    def draw_angle(joint_a, joint_b, joint_c, keypoints, frame, vis_threshold = 0.7):
        height, width = frame.shape[:2]

        a_coords, a_pres, a_vis = joint_a
        b_coords, b_pres, b_vis = joint_b
        c_coords, c_pres, c_vis = joint_c

        # discard z value
        if len(a_coords) > 2: a_coords = a_coords[:2]
        if len(b_coords) > 2: b_coords = b_coords[:2]
        if len(c_coords) > 2: c_coords = c_coords[:2]

        if a_vis > vis_threshold:
            cv2.circle(frame, Tools.int_tuple(a_coords), 6, Graphics.grey, -1)

        if b_vis > vis_threshold:
            cv2.circle(frame, Tools.int_tuple(b_coords), 6, Graphics.grey, -1)

        if c_vis > vis_threshold:
            cv2.circle(frame, Tools.int_tuple(c_coords), 6, Graphics.grey, -1)

        if a_vis > vis_threshold and b_vis > vis_threshold:
            cv2.line(frame, Tools.int_tuple(a_coords), Tools.int_tuple(b_coords), Graphics.pink, 3)

        if b_vis > vis_threshold and c_vis > vis_threshold:
            cv2.line(frame, Tools.int_tuple(b_coords), Tools.int_tuple(c_coords), Graphics.pink, 3)

        if all([a_vis > 0.8, b_vis > 0.8, c_vis > 0.8]):
            cv2.putText(frame, f"{round(Graphics.get_angle_2d(Graphics.make_euclidean(a_coords, height, width), Graphics.make_euclidean(b_coords, height, width), Graphics.make_euclidean(c_coords, height, width)), 2)}", Tools.int_tuple(b_coords), cv2.FONT_HERSHEY_SIMPLEX, 1, Graphics.green, 2)


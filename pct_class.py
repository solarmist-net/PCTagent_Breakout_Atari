"""
Created on Wed Jul 28 00:37:11 2021

@author: gtaus
"""

from typing import Deque, List, Tuple

import collections
import time

import cv2
import gym
import numpy as np

from consts import CROP_HEIGHT, CROP_WIDTH, X_CROP, Y_CROP

# class pct(object):


def angle_between(
    center_x: float, center_y: float, touch_x: float, touch_y: float
) -> float:
    delta_x = touch_x - center_x
    delta_y = center_y - touch_y
    theta_radians = np.arctan2(delta_y, delta_x)
    return np.rad2deg(theta_radians)


def slope(P11x: int, P11y: int, P22x: int, P22y: int) -> float:

    try:
        M = (P22y - P11y) / (P22x - P11x)
    except ZeroDivisionError:
        M = 0
    return M


def y_intercept(P1x: int, P1y: int, slope: float) -> float:
    return P1y - slope * P1x


def line_intersect(m1: float, b1: int, m2: float, b2: int) -> Tuple[float, float]:
    if m1 == m2:
        return 5, Y_CROP
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y


def line_intersection(
    line1: Tuple[Tuple[int, int], Tuple[int, int]],
    line2: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Tuple[float, float]:
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return 0, 0
    else:
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y


def generate_noise_set(x: List[float], constant=5.0) -> List[float]:
    """Use properties of floats to create noise."""
    x[0] = 0.5 - np.random.rand()
    x[1] = x[1] + (x[0] - x[1]) / constant
    x[2] = x[2] + (x[1] - x[2]) / constant
    x[3] = x[3] + (x[2] - x[3]) / constant
    x[4] = x[4] + (x[3] - x[4]) / constant
    return x


def noise(max_size: int = 800) -> Deque[float]:
    constant = 5.0
    half_range = list(range(max_size // 2 - 1))
    full_range = list(range(max_size - 1))

    x: List[float] = [0, 0, 0, 0, 0]
    maxx = 0
    d = collections.deque([0.0] * max_size, maxlen=max_size)

    for _ in half_range:
        x = generate_noise_set(x, constant)

    for i in full_range:
        x = generate_noise_set(x, constant)
        latest = 10_000 * x[4]
        d.append(latest)
        maxx = max(maxx, np.abs(latest))

    for i in full_range:
        # Percent of maxx
        d[i] = d[i] / (maxx) * 100.0

    return d


def img_proc(img):

    # Crop Image
    y_cols = slice(Y_CROP, Y_CROP + CROP_HEIGHT)
    x_cols = slice(X_CROP, X_CROP + CROP_WIDTH)
    img_ball = img[y_cols, x_cols]

    # Image Processing PCT - Convert it to Grayscale
    grayscale_ball = cv2.cvtColor(img_ball, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(grayscale_ball, 100, 255, cv2.THRESH_OTSU)
    bin_ball = binary
    ## Find Contours and Start Algo
    contours, _ = cv2.findContours(bin_ball, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # x0, y0, w0, h0 = cv2.boundingRect(contours_ball[0])
    # To Reset the Game with One Contour Only
    return contours


def ray_tracing(x_ball, y_ball, x_ball_1, y_ball_1):

    # A1, A2 = [x_ball, y_ball], [x_ball_1,y_ball_1]
    # B1, B2 = [-300,Y_CROP], [300,Y_CROP]
    # Points on Horizontal Line
    slope_A = slope(x_ball, y_ball, x_ball_1, y_ball_1)
    slope_B = slope(-300, Y_CROP, 300, Y_CROP)
    y_int_A = y_intercept(x_ball, y_ball, slope_A)
    y_int_B = y_intercept(-300, Y_CROP, slope_B)
    B = line_intersect(slope_A, y_int_A, slope_B, y_int_B)
    # B11 = line_intersection((B1,B2), (A1,A2))
    return B


def bbox(thing):
    # thing = np.asarray(thing)
    bbox_plate = cv2.boundingRect(thing)
    x_plate, y_plate, w_plate, h_plate = bbox_plate
    # Paddle Coordinates
    x_cent_plate = float(x_plate + (w_plate / 2))
    y_cent_plate = float(y_plate)
    return x_cent_plate, y_cent_plate, w_plate

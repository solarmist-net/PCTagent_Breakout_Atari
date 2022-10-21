#!/usr/bin/env python
"""
Created on Fri Jun 25 01:12:23 2021
@author: Tauseef Gulrez and Warren Mansell
Melbourne, Australia and Manchester, UK

Created on Sat Jun  5 23:11:13 2021
@author: gtaus

"""
from typing import Optional, Tuple, cast

import collections
import sys
import time

import cv2
import gym
import numpy as np

from gym.spaces import Discrete

from consts import (
    ACTIONS,
    ASCII_DASH,
    ASCII_SPACE,
    BLUE,
    CENTER_OFFSET,
    CROP_HEIGHT,
    CROP_WIDTH,
    FILL,
    GREEN,
    IMG_HEIGHT,
    IMG_WIDTH,
    RENDER_MODE,
    UTF_DASH_FULL,
    X_CROP,
    Y_CROP,
)

args = "BreakoutNoFrameskip-v4" if len(sys.argv) < 2 else sys.argv[1]
env = gym.make(args, render_mode=RENDER_MODE, full_action_space=True)

# Determine the valid actions
NUM_ACTIONS = cast(Discrete, env.action_space).n
RESTART_KEYS = (UTF_DASH_FULL, ASCII_DASH)
PAUSE_KEYS = (ASCII_SPACE,)

# Human interactions
SKIP_CONTROL = 0
human_wants_restart = False
human_set_pause = False
human_agent_action: Optional[ACTIONS] = None


# Game Dynamics List of Ball and Paddle Positions
MAX_REWARDS = 100
rewards = collections.deque([0] * MAX_REWARDS, maxlen=MAX_REWARDS)
x_balli = collections.deque([0.0] * 2, maxlen=2)
y_balli = collections.deque([0.0] * 2, maxlen=2)
x_platei = collections.deque([0.0] * 2, maxlen=2)
y_platei = collections.deque([0.0] * 2, maxlen=2)


def add_shapes_for_contour(img, x: int, y: int, w: int, h: int):
    # Draw a rectangle around the contour
    cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, thickness=1)
    cent_x, cent_y = round(x + w / 2), round(y)
    # Draw a circle around the contour
    cv2.circle(img, (cent_x, cent_y), radius=1, color=BLUE, thickness=FILL)


def update_position(contour, x_list, y_list) -> Tuple[int, int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour)
    # print(x, y, w, _h)
    center = x + (w / 2)
    x_platei.append(float(center))
    y_platei.append(float(y))
    return x, y, w, h, center


def update_plate_position(contour) -> Tuple[int, int, int, int, int]:
    return update_position(contour, x_platei, y_platei)


def update_ball_position(contour) -> Tuple[int, int, int, int, int]:
    return update_position(contour, x_balli, y_balli)


def process_frame(img, itr: int, has_reward: bool) -> Tuple[bool, int, ACTIONS]:
    """Determines the position of the ball and paddle then decides on an action to take.

    :param img: A frame of gameplay

    :return: Is the game over?
    """
    agent_action = ACTIONS.NOOP
    # Image Processing - the full image

    # Cropped Image: For Ball to detect crop Image Just under the Bricks
    y_cols = slice(Y_CROP, Y_CROP + CROP_HEIGHT)
    x_cols = slice(X_CROP, X_CROP + CROP_WIDTH)
    img_ball = img[y_cols, x_cols]
    # Image Processing PCT - Convert it to Grayscale
    grayscale_ball = cv2.cvtColor(img_ball, cv2.COLOR_BGR2GRAY)
    _ret, binary = cv2.threshold(grayscale_ball, CROP_HEIGHT, 255, cv2.THRESH_OTSU)
    bin_ball = binary
    ## Find Contours and Start Algo
    contours, _hierarchy = cv2.findContours(
        bin_ball,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    num_contours = len(contours)
    cnt_plate: Optional[Tuple[int, int, int, int]] = None
    cnt_ball: Optional[Tuple[int, int, int, int]] = None
    cnt_plate, cnt_ball = None, None
    if num_contours == 0:
        print("Nothing is Detected Something Wrong")
        return True, itr, ACTIONS.NOOP
    elif num_contours == 1:
        cnt_plate = contours[0]
    else:
        cnt_plate, cnt_ball = contours
        add_shapes_for_contour(img_ball, *cv2.boundingRect(cnt_ball))
    add_shapes_for_contour(img_ball, *cv2.boundingRect(cnt_plate))

    # To Reset the Game with One Contour Only
    # There is no ball
    if num_contours == 1:
        ## Get the Plate's bounding rect
        update_plate_position(cnt_plate)
        return False, itr, ACTIONS.FIRE

    x_ball, y_ball, *_ = update_ball_position(cnt_ball)
    _, _, w_plate, _, cent_plate = update_plate_position(cnt_plate)

    # Perceptual Ray Tracing
    # Parameters
    # A1, A2 = [x_ball, y_ball], [x_balli[0], y_balli[0]]
    cv2.line(img_ball, [-300, Y_CROP], [300, Y_CROP], BLUE, 1)
    cv2.line(img_ball, [CROP_WIDTH - 1, Y_CROP], [CROP_WIDTH - 1, 0], BLUE, 1)
    cv2.line(img_ball, [0, Y_CROP], [0, -Y_CROP], BLUE, 1)
    ball_base = x_ball
    # Distance Between Ball and the Plate
    dist_vib = w_plate / 2
    cpoint_dist = dist_vib - 0.25

    # Green Dot on the Line
    cv2.circle(
        img_ball,
        (round(x_ball), Y_CROP),
        radius=1,
        color=GREEN,
        thickness=-1,
    )
    imgS = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    cv2.imshow("pctAgent", imgS)

    if not has_reward:
        return False, itr, ACTIONS.NOOP

    # Negative is down, positive is up
    ball_y_direction = (Y_CROP - y_ball) - (Y_CROP - y_balli[0])
    if ball_y_direction < 0:  # If Ball going Down
        # Hiearchical Loop
        # First Perceptual Reference
        R1 = 0
        # All Gains
        k1, k2, k3, k4 = -1, 1, 1, 1
        # Distance Control
        D = abs(ball_base - cent_plate) + 5
        distance_error = R1 - D
        R2 = distance_error * k1
        # Movement Control
        MD = np.sign(ball_base - cent_plate)
        sign = 1
        if MD < 0:
            sign, agent_action = -1, ACTIONS.LEFT
        else:
            agent_action = ACTIONS.RIGHT
        movement_error = R2 - MD
        ref_position = movement_error * k2
        # Position Control
        position_error = ref_position + sign * cent_plate
        ref_button_press = position_error * k3
        button_press_error = ref_button_press - sign * ball_base
        BP = button_press_error * k4

        if MD != 0:
            itr += 1
        if (
            itr > 1
            or BP == ball_base
            or (R2 < cpoint_dist)
            or (MD > 0 and cent_plate > 136)
        ):
            agent_action = ACTIONS.NOOP
            itr = 0

    # Strategy to come in the Middle - Otherwise include a Velocity Control Reference
    elif ball_y_direction > 0:  # If Ball Going up Come in the Center
        disp = ball_base - cent_plate
        agent_action = ACTIONS.LEFT if disp < 0 else ACTIONS.RIGHT
        if disp != 0:
            itr += 1
        if itr > 1 or abs(CENTER_OFFSET - cent_plate) < dist_vib:
            # if itr > 1 or abs(base - x_cent_plate) < dist_vib:
            agent_action = ACTIONS.NOOP
            itr = 0

    return False, itr, agent_action


def rollout(env):
    global human_agent_action, human_wants_restart, human_set_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    itr = 0
    # Game's Main Loop

    action = ACTIONS.NOOP
    while 1:
        if skip > 0:
            skip -= 1
        else:
            if human_agent_action is not None:
                action = human_agent_action
                human_agent_action = None
            if action != ACTIONS.NOOP:
                # print(f"taking action {action}")
                pass
            total_timesteps += 1
            skip = SKIP_CONTROL

        observation, reward, terminated, truncated, info = env.step(action.value)
        total_reward += reward
        # Put latest 100 rewards in a list
        rewards.append(total_reward)
        # print(f"reward {total_reward}")
        window = env.render()
        done, itr, action = process_frame(window, itr, total_reward > -1)

        # ------------OLD CODE-------------#
        if terminated or done or human_wants_restart:
            break
        while human_set_pause:
            env.render()
            time.sleep(0.1)

        time.sleep(0.001)
    print(f"timesteps {total_timesteps} reward {total_reward}")


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_set_pause
    if key in RESTART_KEYS:
        human_wants_restart = True
    if key in PAUSE_KEYS:  # Toggle pause
        human_set_pause = not human_set_pause
    action = int(key - ASCII_0)
    if action not in ACTIONS:
        return
    human_agent_action = action


def key_release(key, mod):
    global human_agent_action
    action = int(key - ASCII_0)
    if action not in ACTIONS:
        return
    if human_agent_action == action:
        human_agent_action = ACTIONS.NOOP


def main():

    if not hasattr(env.action_space, "n"):
        raise Exception("Keyboard agent only supports discrete action spaces")

    env.reset()
    env.render()
    # Set key handlers
    # env.unwrapped.viewer.window.on_key_press = key_press  # type: ignore
    # env.unwrapped.viewer.window.on_key_release = key_release  # type: ignore

    print(f"ACTIONS={ACTIONS}")
    print(
        f"Press keys {[a.value for a in ACTIONS]} ... to take actions {list(ACTIONS)} ..."
    )
    print("No keys pressed is taking action 0")

    no_of_games = 0
    while 1:
        no_of_games += 1
        window_still_open = rollout(env)
        print(no_of_games)
        if window_still_open is False:
            # outF.close()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()

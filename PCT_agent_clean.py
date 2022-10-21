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
    ASCII_0,
    ASCII_DASH,
    ASCII_SPACE,
    RENDER_MODE,
    UTF_DASH_FULL,
    Y_CROP,
)

# Image Processing Needs to be improved
from pct_class import bbox, img_proc, ray_tracing
from percept_double import percept_double
from percept_double_up import percept_double_up
from percept_multi import percept_multi

RESTART_KEYS = (UTF_DASH_FULL, ASCII_DASH)
PAUSE_KEYS = (ASCII_SPACE,)
TIME_STEP = 0.01
NUM_ACTIONS = 0
SKIP_CONTROL = 0
human_agent_action: Optional[ACTIONS] = None
human_wants_restart = False
human_sets_pause = False


def key_press(key: str, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    assert len(key) == 1
    human_wants_restart = key in RESTART_KEYS
    if key in PAUSE_KEYS:
        human_sets_pause = not human_sets_pause
    action = int(ord(key[0]) - ASCII_0)
    if action not in list(range(NUM_ACTIONS)):
        return
    human_agent_action = ACTIONS(action)


def key_release(key: str, mod):
    global human_agent_action
    assert len(key) == 1
    action = int(ord(key[0]) - ASCII_0)
    if action not in list(range(NUM_ACTIONS)):
        return
    if human_agent_action == action:
        human_agent_action = ACTIONS.NOOP


# Saving Paddle and Ball Positions
x_balli = collections.deque([0.0] * 2, maxlen=2)
y_balli = collections.deque([0.0] * 2, maxlen=2)
x_platei = collections.deque([0.0] * 2, maxlen=2)
y_platei = collections.deque([0.0] * 2, maxlen=2)


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


from visual_perception import detect_edges, find_objects, find_movement


def process_frame(img, itr: int, has_reward: bool) -> Tuple[bool, int, ACTIONS]:
    movements = find_movement(img, itr, TIME_STEP)
    if movements:
        print([m.vel for m in movements if m.vel])

    agent_action = ACTIONS.NOOP
    # Image Processing of the Environment
    contour_ball = None
    try:
        contour_plate, *contours = img_proc(img)
    except (ValueError) as err:
        print("Nothing is Detecte Something Wrong")
        return True, itr, agent_action

    # Generate Points of the Detected Objects
    x_cent_plate, y_cent_plate, w_plate = bbox(contour_plate)
    x_platei.append(x_cent_plate)
    y_platei.append(y_cent_plate)

    try:
        contour_ball = contours[0] if isinstance(contours, list) else None
    except IndexError as err:
        agent_action = ACTIONS.FIRE
        return False, itr, agent_action

    # Store everything as a two array matrix
    x_ball, y_ball, w_ball = bbox(contour_ball)
    x_balli.append(x_ball)
    y_balli.append(y_ball)

    oldes_x, oldest_y = x_balli[0], y_balli[0]
    # # Perceptual Ray Tracing
    B = ray_tracing(x_ball, y_ball, oldes_x, oldest_y)
    # Distance Between Ball and the Plate
    cpoint_dist = (w_plate / 2) - 0.25
    # Ball and Paddle Previous Positions
    bx = B[0]
    fx = x_cent_plate
    # First Perceptual Input
    R1 = 0
    dist_vib = w_plate / 2
    dist_vib1 = 6

    if not has_reward:
        return False, itr, agent_action

    ball_y_direction = (Y_CROP - y_ball) - (Y_CROP - oldest_y)

    bx1 = 70
    if ball_y_direction < 0:  # If Ball going Down
        # Multi-Hierarchical PCT Mode
        # human_agent_action, itr = percept_multi(bx,fx,itr,R1,cpoint_dist,k1,k2,k3,k4)
        # Double-Hierarchical Mode (Requires T=Ray Tracing)
        agent_action, itr = percept_double_up(bx, fx, itr, dist_vib1)

    # Image Processing Needs to be done
    if ball_y_direction > 0:  # If Ball Going up Track it
        agent_action, itr = percept_double(bx1, itr, dist_vib, x_cent_plate)

    return False, itr, agent_action


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    itr = 0

    # Game's Main Loop
    action = ACTIONS.NOOP
    # All Gains
    k1, k2, k3, k4 = -1, 1, 1, 1

    while 1:
        if skip:
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
        obser, reward, terminated, truncated, info = env.step(action.value)

        total_reward += reward
        # window_still_open = env.render()
        done, itr, action = process_frame(env.render(), itr, total_reward > -1)

        # if not isinstance(window_still_open, np.ndarray):
        #     breakpoint()
        #     return False
        if terminated or done or human_wants_restart:
            break
        while human_sets_pause:
            env.render()
            time.sleep(TIME_STEP * 10)

        time.sleep(TIME_STEP)
    print(f"timesteps {total_timesteps} reward {total_reward}")


def main(args):
    global NUM_ACTIONS
    env = gym.make(args, render_mode=RENDER_MODE, full_action_space=True)

    if not hasattr(env.action_space, "n"):
        raise Exception("Keyboard agent only supports discrete action spaces")
    NUM_ACTIONS = cast(Discrete, env.action_space).n

    env.reset()
    env.render()
    # env.unwrapped.viewer.window.on_key_press = key_press
    # env.unwrapped.viewer.window.on_key_release = key_release
    print(f"ACTIONS={ACTIONS}")
    print(
        f"Press keys {[a.value for a in ACTIONS]} ... to take actions {list(ACTIONS)} ..."
    )
    print("No keys pressed is taking action 0")

    no_of_games = 0

    while 1:
        no_of_games = no_of_games + 1
        window_still_open = rollout(env)
        print(f"Games played: {no_of_games}")
        if window_still_open == False:
            # outF.close()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    # To save Rewards
    # outF = open("results_Breakout-V4_500.txt", "a")
    args = "BreakoutNoFrameskip-v4" if len(sys.argv) < 2 else sys.argv[1]
    main(args)

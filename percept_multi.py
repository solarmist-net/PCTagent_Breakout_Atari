"""
Created on Wed Jul 28 17:18:33 2021

@author: gtaus
"""

from typing import Tuple
import numpy as np

from consts import ACTIONS, FX_LIMIT


def percept_multi(bx, fx, itr, R1, cpoint_dist, k1, k2, k3, k4) -> Tuple[ACTIONS, int]:
    agent_action = ACTIONS.NOOP
    # First Perceptual Rference
    # Distance Control
    D = abs(bx - fx) + 5  # NoFrameskipe 405
    distance_error = R1 - D
    R2 = distance_error * k1
    # Movement Control
    MD = np.sign(bx - fx)
    movement_error = R2 - MD
    sign = -1 if MD < 0 else 1
    agent_action = ACTIONS.LEFT if MD < 0 else ACTIONS.RIGHT
    R3 = movement_error * k2

    # Position Control
    position_error = R3 + sign * fx
    R4 = position_error * k3
    button_press_error = R4 - sign * bx
    BP = button_press_error * k4

    if MD != 0:
        itr += 1
    if itr > 1 or BP == bx or (R2 < cpoint_dist) or (MD > 0 and fx > FX_LIMIT):
        agent_action = ACTIONS.NOOP
        itr = 0

    return agent_action, itr

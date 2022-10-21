"""
Created on Wed Jul 28 17:43:05 2021

@author: gtaus
"""
from typing import Tuple

from consts import ACTIONS


def percept_double(bx: int, itr: int, dist_vib, x_cent_plate) -> Tuple[ACTIONS, int]:
    agent_action = ACTIONS.NOOP
    disp = bx - x_cent_plate
    agent_action = ACTIONS.LEFT if disp < 0 else ACTIONS.RIGHT
    if disp != 0:
        itr += 1
    if itr > 1 or abs(disp) < dist_vib:
        agent_action = ACTIONS.NOOP  # 0
        itr = 0

    return agent_action, itr

"""
Created on Wed Jul 28 20:21:02 2021

@author: gtaus
"""
from typing import Tuple

from consts import ACTIONS, FX_LIMIT


def percept_double_up(bx, fx, itr, dist_vib) -> Tuple[ACTIONS, int]:
    disp = bx - fx
    agent_action = ACTIONS.LEFT if (disp) < 0 else ACTIONS.RIGHT
    if disp != 0:
        itr += 1
    if itr > 1 or abs(disp) < dist_vib or fx > FX_LIMIT:
        agent_action = ACTIONS.NOOP
        itr = 0

    return agent_action, itr

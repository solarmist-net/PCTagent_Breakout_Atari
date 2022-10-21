from enum import Enum


# Atari specific
# These are all the possible actions in the games
class ACTIONS(Enum):
    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    UPRIGHT = 6  # Up + Right
    UPLEFT = 7  # Up + Left
    DOWNRIGHT = 8  # Down + Right
    DOWNLEFT = 9  # Down + Left
    UPFIRE = 10  # Up + Fire
    RIGHTFIRE = 11  # Right + Fire
    LEFTFIRE = 12  # Left + Fire
    DOWNFIRE = 13  # Down + Fire
    UPRIGHTFIRE = 14  # Up + Right + Fire
    UPLEFTFIRE = 15  # Up + Left + Fire
    DOWNRIGHTFIRE = 16  # Down + Right + Fire
    DOWNLEFTFIRE = 17  # Down + Left + Fire


# Constants
FILL = -1
RENDER_MODE = "rgb_array"
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Center Point for Ball to Maintain
CENTER_OFFSET = 70
FX_LIMIT = 136
# Crop width and height
CROP_HEIGHT, CROP_WIDTH = 100, 141

# Keys
ASCII_0 = ord("0")
ASCII_SPACE = ord(" ")
ASCII_SPACE = ord(" ")
ASCII_DASH = ord("-")
UTF_DASH_FULL = ord("－")

# Img size
IMG_WIDTH, IMG_HEIGHT = 400, 700
# Ball X and Y Crop
Y_CROP, X_CROP = 94, 9

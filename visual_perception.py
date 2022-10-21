from typing import Set, Tuple, Dict, Optional
import cv2
import numpy as np
import matplotlib.image
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int

    def __hash__(self) -> int:
        return hash(self.tuple)

    def __repr__(self) -> str:
        return f"Pt({self.x}, {self.y})"

    def __lt__(self, other: "Point") -> bool:
        return self.tuple < other.tuple

    @property
    def tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Color:
    r: int
    g: int
    b: int

    def __hash__(self) -> int:
        return hash(self.tuple)

    def __repr__(self) -> str:
        return f"Color(r={self.r},g={self.g},b={self.b})"

    @property
    def tuple(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)


@dataclass
class ScreenObject:
    pos: Point
    color: Color
    area: int
    width: int
    height: int
    vel: Optional[Tuple[float, float]] = None

    def __lt__(self, other: "Point") -> bool:
        return self.pos < other.pos

    def __hash__(self) -> int:
        return hash(
            (self.pos, self.color, self.area, self.width, self.height, self.vel)
        )

    def __repr__(self) -> str:
        return f"ScreenObject(pos={self.pos}, color={self.color}, area={self.area})"


BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)


def motion_mask(img1, img2: np.ndarray) -> np.ndarray:
    diff = np.abs(img1 - img2)

    return np.ma.masked_not_equal(diff, BLACK.tuple).mask


def img_jitter_mask(img: np.ndarray, first=False, column=False) -> np.ndarray:
    """Detect edges using jitter."""
    axis = 0 if column else 1
    idx = 0 if first else -1
    opp = -1 if first else 0
    img_j = np.delete(img, idx, axis=axis)
    img_j = np.insert(img_j, opp, values=0, axis=axis)

    return motion_mask(img, img_j)


def detect_edges(img: np.ndarray, itr: int, debug=False) -> np.ndarray:
    """Find edges using jitter.
    Stereo vision combined with eye movement lets you detect objects by volume.
    """
    mask = img_jitter_mask(img, first=True, column=True)
    mask += img_jitter_mask(img, first=False, column=True)
    mask += img_jitter_mask(img, first=True)
    mask += img_jitter_mask(img, first=False)

    edges = img * mask

    if debug:
        matplotlib.image.imsave(f"frames/frame{itr}_res.png", edges)
        matplotlib.image.imsave(f"frames/frame{itr}_o.png", img)

    return edges


def find_objects(
    img: np.ndarray,
) -> Set[ScreenObject]:
    """Find objects in image."""
    colors = {
        Color(*c) for row in img for c in np.unique(row, axis=0) if Color(*c) != BLACK
    }
    # Find objects by color

    objs = set()
    # Assumes objects stay the same color.
    # If not, then we need to find the object by movement.
    for color in colors:
        # Mask everything not the color
        mask = np.ma.masked_not_equal(img, color.tuple).filled(BLACK.tuple)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Find contours
        _ret, binary = cv2.threshold(mask, 100, 255, cv2.THRESH_OTSU)
        contours, _hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # Find bounding boxes for each contour
        for contour in contours:
            rect = cv2.boundingRect(contour)
            pos = Point(rect[0], rect[1])
            area = np.count_nonzero(
                mask[pos.y : pos.y + rect[3], pos.x : pos.x + rect[2]]
            )
            objs.add(ScreenObject(pos, color, area, width=rect[2], height=rect[3]))

    return objs


def dist(p1: Point, p2: Point) -> float:
    return np.linalg.norm(np.array(p1.tuple) - np.array(p2.tuple))


def start_movement(
    initial: Optional[np.ndarray] = None,
):
    """Find movement between two images."""
    previous = initial
    prev_edges = np.ndarray((0, 0, 0))
    prev_objs = {}

    def find_movement(
        current: np.ndarray,
        itr=0,
        time=0.01,
    ) -> Set[ScreenObject]:
        """Find velocity, object (acceleration?) pairs."""
        nonlocal previous, prev_edges, prev_objs
        edges = detect_edges(current, itr)
        objects = find_objects(edges)
        if previous is None:
            previous = current
            prev_edges = edges
            prev_objs = objects

        movements = set()
        cur_match_needed = set()
        matched = set()
        for obj in prev_objs:
            if obj in objects:
                matched.add(obj)
            else:
                cur_match_needed.add(obj)

        cur_unmatched = objects - matched
        for obj in cur_match_needed:

            dists = sorted([(dist(obj.pos, cobj.pos), cobj) for cobj in cur_unmatched])
            color_match = {o for o in cur_unmatched if o.color == obj.color}
            size_match = {o for o in cur_unmatched if o.area == obj.area}
            # If the size and color match
            overlap = size_match.intersection(color_match)
            # Find the object (by size and position) in the current frame
            if len(overlap) == 1 and dists[0][1] in overlap:
                obj.vel = (
                    (dists[0][1].pos.x - obj.pos.x) / time,
                    (dists[0][1].pos.y - obj.pos.y) / time,
                )
                movements.add(obj)
            else:
                # Unmatched movement
                matplotlib.image.imsave(f"frames/frame{itr}_cur.png", current)
                matplotlib.image.imsave(f"frames/frame{itr}_prev.png", previous)
                matplotlib.image.imsave(
                    f"frames/frame{itr}_diff.png", current - previous
                )
                breakpoint()
                pass

        previous = current
        prev_edges = edges
        prev_objs = objects
        return movements

    return find_movement


find_movement = start_movement()

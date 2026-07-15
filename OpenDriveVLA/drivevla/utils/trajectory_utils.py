import re
from typing import List, Tuple


def split_reason_traj(text: str) -> Tuple[str, str]:
    """Split a reason+traj assistant output into (reasoning, trajectory_span).

    Why this is mandatory before retrieve_traj: retrieve_traj strips all letters and
    parses whatever digits remain as waypoints. The reasoning text is full of digits
    ("about 5.7 m away", "last 2 seconds") -- feed it the whole output and those numbers
    become SPURIOUS waypoints mixed into the real trajectory, silently corrupting every L2.
    So isolate the <traj_start>..<traj_end> span first.

    Robust to: trailing special tokens (<|im_end|> etc.), and legacy trajectory-only output
    with no reason/traj markers at all (returns ("", text) so old checkpoints still parse)."""
    reason = ""
    m = re.search(r"<reason_start>(.*?)<reason_end>", text, re.S)
    if m:
        reason = m.group(1).strip()
    m = re.search(r"<traj_start>(.*?)<traj_end>", text, re.S)
    if m:
        return reason, m.group(1)
    # No closed traj block: strip any reasoning, hand back the rest (covers a truncated
    # generation, or a trajectory-only legacy model that never emitted the markers).
    traj = re.sub(r"<reason_start>.*?<reason_end>", "", text, flags=re.S)
    return reason, traj


def retrieve_traj(text: str) -> List[Tuple[float, float]]:
    """Retrieve the trajectory from the output."""
    # Remove all English letters from text
    text = re.sub(r'[a-zA-Z]', '', text)

    # Remove Chinese characters from text
    text = re.sub(r'[\u4e00-\u9fff]', '', text)

    # Fix numbers with consecutive decimal points by keeping only first decimal point
    text = re.sub(r'(\d+)\.\.+', r'\1.', text)

    # Remove < and > symbols from text
    text = re.sub(r'[<>]', '', text)

    # add missing comma between numbers in coordinates `(x y)` → `(x, y)`
    text = re.sub(r'(\d+\.\d+)\s+(\d+\.\d+)(?=\s*[,\)])', r'\1, \2', text)

    # Fix numbers with minus sign in the middle by keeping both numbers
    text = re.sub(r'(\d+\.\d+)-(\d+\.\d+)', r'\1, \2', text)

    # remove extra numbers in the middle of coordinates `(x, y, z)` → `(x, y)`
    text = re.sub(r'(\(\s*-?\d+\.\d+,\s*-?\d+\.\d+),\s*-?\d+\.\d+(\s*\))', r'\1\2', text)

    # Remove all spaces in text
    text = re.sub(r'\s+', '', text)

    # Fix numbers with multiple decimal points by keeping only first decimal point
    text = re.sub(r'(\d+\.\d+)\.(\d+)', r'\1\2', text)

    # Fix numbers with consecutive decimal points by keeping only first decimal point
    text = re.sub(r'(\d+)\.\.(\d+)', r'\1.\2', text)

    # Fix numbers with minus sign after decimal point by removing the minus sign
    text = re.sub(r'(\d+)\.\-(\d+)', r'\1.\2', text)

    coord_pairs = re.findall(r'[\[\(]([-\deE.+]+),\s*([-\deE.+]+)[\]\)]', text)
    coords_list = [(float(x), float(y)) for x, y in coord_pairs]
    if len(coords_list) < 6:
        for i in range(6 - len(coords_list)):
            coords_list.append(coords_list[-1])
    elif len(coords_list) > 6:
        coords_list = coords_list[:6]
    return coords_list

def trajectory_is_valid(trajectory: List[Tuple[float, float]]) -> bool:
    """Check if the trajectory is of type list[tuple[float, float]] and has a length of 6."""
    return isinstance(trajectory, list) \
            and all(isinstance(i, tuple) and len(i) == 2 and all(isinstance(coord, float) for coord in i) for i in trajectory) \
            and len(trajectory) == 6

def check_traj(trajectory: List[Tuple[float, float]]) -> None:
    assert trajectory_is_valid(trajectory), f"VLM Output Trajectory is not valid: {trajectory}"
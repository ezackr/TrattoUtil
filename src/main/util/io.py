import os


def get_parent_dir(path: str, level: int = 1) -> str:
    """
    Returns the (level)-th parent directory of a given path
    :param path: the root path
    :param level: the number of parent directories above the root path
    :return: the (level)-th parent directory of the given path
    :raise ValueError: if the given level is less than 0
    """
    if level == 0:
        return path
    if level < 0:
        raise ValueError(f"Parent directory level must be greater than 0. Actual: {level}")
    for _ in range(level):
        path = os.path.dirname(path)
    return path


current_path: str = os.path.abspath(__file__)
root_dir: str = get_parent_dir(current_path, level=4)

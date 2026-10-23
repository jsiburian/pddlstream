"""Helpers shared by the PyBullet examples."""
from examples.pybullet.utils.pybullet_tools.utils import joint_from_name


def get_custom_limits(robot, base_limits, yaw_limit=None):
    """Joint limits for a mobile base with x, y and theta joints, given ((x_min, y_min), (x_max, y_max))."""
    x_limits, y_limits = zip(*base_limits)
    custom_limits = {
        joint_from_name(robot, 'x'): x_limits,
        joint_from_name(robot, 'y'): y_limits,
    }
    if yaw_limit is not None:
        custom_limits.update({
            joint_from_name(robot, 'theta'): yaw_limit,
        })
    return custom_limits

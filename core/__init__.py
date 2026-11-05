"""Shared ROS-free core: the KITTI data layer and the pose math.

One copy of this package goes into both images. It imports only numpy and the
standard library, because the publisher image has no scipy.
"""

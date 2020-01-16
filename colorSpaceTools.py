"""A set of functions to help with converting color spaces. Assumes RGB are apple RGB w/ D65 white point"""
import numpy as np

#Source: http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
apple_rgb_to_xyz_matrix = np.array([[0.4497288, 0.3162486, 0.1844926], [0.2446525, 0.6720283, 0.0833192], [0.0251848, 0.1411824, 0.9224628]])
epsilon = 216 / 24389
kappa = 24389 / 27

#2 degree observer
d65_XYZ = np.array([0.95047, 1.0, 1.08883])

X = 0
Y = 1
Z = 2

def rgb_to_xyz(rgb):
    """Convert RGB color value to XYZ color value"""
    return np.matmul(apple_rgb_to_xyz_matrix, rgb)

def xyz_to_lab(XYZ):
    """Convert XYZ color value to CIE LAB color value"""
    xyz_scaled = XYZ / d65_XYZ

    epsilon_mask = xyz_scaled > epsilon

    xyz_processed = ((xyz_scaled * kappa) + 16) / 116
    xyz_processed[epsilon_mask] = np.cbrt(xyz_scaled)[epsilon_mask]

    L = (116 * xyz_processed[Y]) - 16
    a = (xyz_processed[X] - xyz_processed[Y]) * 500
    b = (xyz_processed[Y] - xyz_processed[Z]) * 200

    return np.array([L, a, b])

def rgb_to_lab(rgb):
    """Convert RGB color value to CIE LAB color value"""
    xyz = rgb_to_xyz(rgb)
    return xyz_to_lab(xyz)

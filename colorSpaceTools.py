import numpy as np

#adobe_rgb_to_xyz_matrix = np.array([[0.5767309, 0.1855540, 0.1881852], [0.2973769, 0.6273491, 0.0752741], [0.0270343, 0.0706872, 0.9911085]])
apple_rgb_to_xyz_matrix = np.array([[0.4497288, 0.3162486, 0.1844926], [0.2446525, 0.6720283, 0.0833192], [0.0251848, 0.1411824, 0.9224628]])

def rgb_to_xyz(rgb):
    return np.matmul(apple_rgb_to_xyz_matrix, rgb)

#Source: http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
epsilon = 216 / 24389
kappa = 24389 / 27
#2 degree observer
#d65_xyz = np.array([0.31271, 0.32902, 0.35827])
#Found direct Source. My adaptation was correct, but better just to use a constant
d65_XYZ = np.array([0.95047, 1.0, 1.08883])

X = 0
Y = 1
Z = 2

def xyz_to_lab(XYZ):
    #Set d65 to y = 1
    #d65_xyz_scale = 1 / d65_xyz[1]
    #xyz_scaled = xyz / (d65_xyz * d65_xyz_scale)
    xyz_scaled = XYZ / d65_XYZ

    epsilon_mask = xyz_scaled > epsilon

    xyz_processed = ((xyz_scaled * kappa) + 16) / 116
    xyz_processed[epsilon_mask] = np.cbrt(xyz_scaled)[epsilon_mask]

    L = (116 * xyz_processed[Y]) - 16
    a = (xyz_processed[X] - xyz_processed[Y]) * 500
    b = (xyz_processed[Y] - xyz_processed[Z]) * 200

    return np.array([L, a, b])

def rgb_to_lab(rgb):
    xyz = rgb_to_xyz(rgb)
    return xyz_to_lab(xyz)


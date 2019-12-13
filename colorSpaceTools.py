import numpy as np

adobe_rgb_to_xyz_matrix = np.array([[0.5767309, 0.1855540, 0.1881852], [0.2973769, 0.6273491, 0.0752741], [0.0270343, 0.0706872, 0.9911085]])

def rgb_to_xyz(rgb):
    return adobe_rgb_to_xyz_matrix * rgb 

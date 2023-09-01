import numpy as np


def _get_mesh(mapX: np.ndarray, mapY: np.ndarray):
    mesh_cell_size = 16
    mesh0 = []
    # Creates subsampled mesh which will be loaded on to device to undistort the image
    for y in range(mapX.shape[0] + 1):  # iterating over height of the image
        if y % mesh_cell_size == 0:
            row_left = []
            for x in range(mapX.shape[1] + 1):  # iterating over width of the image
                if x % mesh_cell_size == 0:
                    if y == mapX.shape[0] and x == mapX.shape[1]:
                        row_left.append(mapY[y - 1, x - 1])
                        row_left.append(mapX[y - 1, x - 1])
                    elif y == mapX.shape[0]:
                        row_left.append(mapY[y - 1, x])
                        row_left.append(mapX[y - 1, x])
                    elif x == mapX.shape[1]:
                        row_left.append(mapY[y, x - 1])
                        row_left.append(mapX[y, x - 1])
                    else:
                        row_left.append(mapY[y, x])
                        row_left.append(mapX[y, x])
            if (mapX.shape[1] % mesh_cell_size) % 2 != 0:
                row_left.append(0)
                row_left.append(0)

            mesh0.append(row_left)

    mesh0 = np.array(mesh0)
    # mesh = list(map(tuple, mesh0))
    return mesh0

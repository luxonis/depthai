import numpy as np

def _get_mesh(mapX: np.ndarray, mapY: np.ndarray):
    meshCellSize = 16
    mesh0 = []
    # print(mapX.shape)
    # Creates subsampled mesh which will be loaded on to device to undistort the image
    for y in range(mapX.shape[0] + 1):  # iterating over height of the image
        if y % meshCellSize == 0:
            rowLeft = []
            for x in range(mapX.shape[1] + 1):  # iterating over width of the image
                if x % meshCellSize == 0:
                    if y == mapX.shape[0] and x == mapX.shape[1]:
                        rowLeft.append(mapY[y - 1, x - 1])
                        rowLeft.append(mapX[y - 1, x - 1])
                    elif y == mapX.shape[0]:
                        rowLeft.append(mapY[y - 1, x])
                        rowLeft.append(mapX[y - 1, x])
                    elif x == mapX.shape[1]:
                        rowLeft.append(mapY[y, x - 1])
                        rowLeft.append(mapX[y, x - 1])
                    else:
                        rowLeft.append(mapY[y, x])
                        rowLeft.append(mapX[y, x])
            if (mapX.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)

            mesh0.append(rowLeft)

    mesh0 = np.array(mesh0)
    # mesh = list(map(tuple, mesh0))
    return mesh0
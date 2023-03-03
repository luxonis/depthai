#!/usr/bin/env python3

import numpy as np
import math

np.set_printoptions(suppress=True)

def rotationMatrixToEulerAngles(R) :
  
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :
 
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
 
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
 
    R = (R_z @ ( R_y @ R_x ))
 
    return R

def stereoRectify(R, T):
    T = T.flatten()
    print(f'Shape of R: {R.shape}')
    print(f'Shape of T: {T.shape}')
    om = rotationMatrixToEulerAngles(R)
    om = om * -0.5
    r_r = eulerAnglesToRotationMatrix(om)
    t = r_r @ T

    idx = 0 if abs(t[0]) > abs(t[1]) else 1

    c = t[idx]
    nt = np.linalg.norm(t)
    uu = np.zeros(t.shape)
    uu[idx] = 1 if c > 0 else -1
    print(f'Shape of t: {t.shape}')
    print(f'Shape of uu: {uu.shape}')

    ww = np.cross(t, uu)
    nw = np.linalg.norm(ww)
    
    if nw > 0:
        scale = math.acos(abs(c)/nt)/nw
        ww = ww * scale
        
    wR = eulerAnglesToRotationMatrix(ww)
    R1 = wR @ np.transpose(r_r)
    R2 = wR @ r_r

    return R1, R2

refR1 = np.array([[ 0.99994761, -0.00390605, -0.0094629 ], 
                [ 0.00391631, 0.99999177, 0.00106613 ], 
                [ 0.00945866, -0.00110314, 0.99995464 ]])
refR2 = np.array([[ 0.99984103, 0.0033367, 0.01751487 ], 
                [ -0.0033177, 0.99999386, -0.00111375 ], 
                [ -0.01751848, 0.00105546, 0.99984598 ]])


R = np.array([[ 0.99960995, -0.00720377, -0.02698262 ],
            [ 0.00726279,  0.99997145,  0.00208996 ],
            [ 0.02696679, -0.00228512,  0.99963373 ]])
T = np.array([-7.51322365, -0.02507336, -0.13161406])

R1, R2 = stereoRectify(R, T)
print("refR1-R1:")
print(refR1-R1)
print("refR2-R2:")
print(refR2-R2)

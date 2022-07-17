# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 13:14:09 2022

@author: jonat
"""

import cv2
import numpy as np

import matplotlib.pyplot as plt


def straighten(img, interpolated=False):
    ny, nx = img.shape[0], img.shape[1]
    source_corners = get_corners(img)

    # Mapping target corners to the source corners
    target_corners = []

    size = np.sqrt(nx**2 + ny**2)
    for i in range(3):
        for j in range(i+1, 4):
            dist = np.sqrt((source_corners[i][0]-source_corners[j][0])**2
                           + (source_corners[i][1]-source_corners[j][1])**2)
            if dist < size:
                size = dist
    dist = int(dist)
    sy, sx = int(1.414*dist), dist

    form = orientation(source_corners, nx, ny)
    if form == 1:
        sy, sx = sx, sy

    C = [(0, 0), (sx-1, 0), (0, sy-1), (sx-1, sy-1)]
    for x, y in source_corners:
        dist = [np.sqrt(x**2 + y**2), np.sqrt((x-nx)**2 + y**2),
                np.sqrt(x**2 + (y-ny)**2), np.sqrt((x-nx)**2 + (y-ny)**2)]
        target_corners.append(C[dist.index(min(dist))])

    # Applying the transform (see source in ReadMe)
    A = np.asarray([[source_corners[0][0], source_corners[0][1], 1, 0, 0, 0,
                     -target_corners[0][0]*source_corners[0][0],
                     -target_corners[0][0]*source_corners[0][1]],
                    
                    [0, 0, 0, source_corners[0][0], source_corners[0][1], 1,
                     -target_corners[0][1]*source_corners[0][0],
                     -target_corners[0][1]*source_corners[0][1]],
                    
                    [source_corners[1][0], source_corners[1][1], 1, 0, 0, 0,
                     -target_corners[1][0]*source_corners[1][0],
                     -target_corners[1][0]*source_corners[1][1]],
                    
                    [0, 0, 0, source_corners[1][0], source_corners[1][1], 1,
                     -target_corners[1][1]*source_corners[1][0],
                     -target_corners[1][1]*source_corners[1][1]],
                    
                    [source_corners[2][0], source_corners[2][1], 1, 0, 0, 0,
                     -target_corners[2][0]*source_corners[2][0],
                     -target_corners[2][0]*source_corners[2][1]],
                    
                    [0, 0, 0, source_corners[2][0], source_corners[2][1], 1,
                     -target_corners[2][1]*source_corners[2][0],
                     -target_corners[2][1]*source_corners[2][1]],
                    
                    [source_corners[3][0], source_corners[3][1], 1, 0, 0, 0,
                     -target_corners[3][0]*source_corners[3][0],
                     -target_corners[3][0]*source_corners[3][1]],
                    
                    [0, 0, 0, source_corners[3][0], source_corners[3][1], 1,
                     -target_corners[3][1]*source_corners[3][0],
                     -target_corners[3][1]*source_corners[3][1]]
                    ])

    UV = np.asarray([target_corners[0][0], target_corners[0][1],
                     target_corners[1][0], target_corners[1][1],
                    target_corners[2][0], target_corners[2][1],
                    target_corners[3][0], target_corners[3][1]])
    UV = UV.reshape((8, 1))
    M = np.dot(np.linalg.inv(A), UV)
    a, b, c, d, e, f, g, h = M

    if len(img.shape) == 3:
        res = np.zeros((sy, sx, 3))
    else:
        res = np.zeros((sy, sx))
    res = res.astype(np.uint8)

    U = np.asarray(list(range(sx)))
    V = np.asarray(list(range(sy)))
    """
    For a point at coordinates (v,u) in the target image, the coordinates of the
    matching pixel in the source image are :
    x = ((v*h-e)*(c-u) - (f-v)*(u*h-b))/((v*h-e)*(u*g-a) - (v*g-d)*(u*h-b))
    y = ((v*g-d)*(c-u) - (f-v)*(u*g-a))/((v*g-d)*(u*h-b) - (v*h-e)*(u*g-a))
    """
    x_list = (np.dot(V[:, None]*h-e, c-U[None]).ravel()
              - np.dot(f-V[:, None], U[None]*h-b).ravel()) / \
             (np.dot(V[:, None]*h-e, U[None]*g-a).ravel()
              - np.dot(V[:, None]*g-d, U[None]*h-b).ravel())
             
    y_list = (np.dot(V[:, None]*g-d, c-U[None]).ravel()
              - np.dot(f-V[:, None], U[None]*g-a).ravel()) / \
             (np.dot(V[:, None]*g-d, U[None]*h-b).ravel() -
              np.dot(V[:, None]*h-e, U[None]*g-a).ravel())

    if interpolated:
        res = interpolation(img, y_list, x_list, sy, sx)
        res = res.astype(int)
    else:
        y_list, x_list = y_list.astype(int), x_list.astype(int)
        v_list = np.asarray([i//sx for i in range(sx*sy)])
        u_list = np.asarray([i % sx for i in range(sx*sy)])
        res[v_list, u_list] = img[y_list, x_list]

    if form:
        res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)

    return res


def orientation(corners, nx, ny):
    """
    From corners, list of positions of corners on image
    and (ny, nx) the dimensions of this image
    Returns orientation of the papersheet on the photo :
        1 for landscape, 0 for portrait
    """
    # Distances to top left corners
    dist_TL = [np.sqrt(x**2 + y**2) for x, y in corners]
    # Point closest to top left corner
    TL = corners[dist_TL.index(min(dist_TL))]

    # Distances to top right corners
    dist_TR = [np.sqrt((x-nx)**2 + y**2) for x, y in corners]
    # Point closest to top right corner
    TR = corners[dist_TR.index(min(dist_TR))]

    # Distances to bottom left corners
    dist_BL = [np.sqrt(x**2 + (y-ny)**2) for x, y in corners]
    # Point closest to bottom left corner
    BL = corners[dist_BL.index(min(dist_BL))]

    orientation = 0
    dist_TL_TR = np.sqrt((TL[0] - TR[0])**2 + (TL[1] - TR[1])**2)
    dist_TL_BL = np.sqrt((TL[0] - BL[0])**2 + (TL[1] - BL[1])**2)

    # If the Top Right corner is closer to the Top Left one
    # rather than to the Bottom Left one, it is a landscape
    if dist_TL_TR > dist_TL_BL:
        orientation = 1

    return orientation


def interpolation(img, Y, X, ty, tx):
    assert len(X) == len(Y), "Sets of coordinates must have same lengths"
    assert len(X) == ty*tx, "Sets of target coordinates must contain the \
                             same number of elements as the target image"

    YF, XF = Y.astype(int), X.astype(int)
    d1 = np.multiply((1-(X-XF)), (1-(Y-YF)))
    d2 = np.multiply((1-(X-XF)), (Y-YF))
    d3 = np.multiply((X-XF), (1-(Y-YF)))
    d4 = np.multiply((X-XF), (Y-YF))
    v_list = np.asarray([i//tx for i in range(tx*ty)])
    u_list = np.asarray([i % tx for i in range(tx*ty)])
    res = np.zeros((ty,tx,3))
    res[v_list, u_list] = np.expand_dims(d1, 1)*img[YF, XF] \
                          + np.expand_dims(d2, 1)*img[YF+1, XF] \
                          + np.expand_dims(d3, 1)*img[YF, XF+1] \
                          + np.expand_dims(d4, 1)*img[YF+1, XF+1]

    return res


def get_corners(img):
    I = img.copy().astype(np.uint8)

    # Grey-scaling the image
    if len(I.shape) == 3:
        G = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    else:
        G = I.copy()

    ny, nx = G.shape

    # Closing to ignore the content of the paper in the Hough Transform
    closed = cv2.morphologyEx(G, cv2.MORPH_CLOSE, np.ones((2*(nx//40)+1, 2*(nx//40)+1)))

    # Binarisation
    n = 2*(nx//120)+1
    binary = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 2*n+1, n//6)

    # Hough transform
    lines = cv2.HoughLines(binary, 1, np.pi/180, nx//4)

    dr = nx//20  # 5% of nx margin for similarity
    dt = np.pi/18  # 10° margin for similarity

    found_lines = []
    i = 0

    # Searching for the 4 borders
    while len(found_lines) != 4 and i < len(lines):
        r, t = lines[i][0]
        treated = False
        for line in found_lines:
            r2, t2 = line
            if r2 < 0:
                r2 = abs(r2)
                t2 -= np.pi
            if r < 0:
                r = abs(r)
                t -= np.pi
            
            if (abs(r - r2) < dr and abs(t - t2) < dt):
                treated = True

        if not(treated):
            found_lines.append(tuple(lines[i][0]))

        i += 1

    # Finding corners
    corners = []  # List of intersections points
    for i in range(3):
        for j in range(i+1, 4):
            r1 = found_lines[i][0]
            t1 = found_lines[i][1]
            r2 = found_lines[j][0]
            t2 = found_lines[j][1]
            if abs(t1 - t2) > np.pi/4:
                if t1 != 0 and t2 != 0:
                    x = np.tan(t1)*np.tan(t2)/(np.tan(t2)-np.tan(t1)) \
                        * (r1/np.sin(t1) - r2/np.sin(t2))
                    y = -x/np.tan(t1) + r1/np.sin(t1)
    
                elif t1 == 0:
                    x = r1
                    y = -x/np.tan(t2) + r2/np.sin(t2)
    
                else:
                    x = r2
                    y = -x/np.tan(t1) + r1/np.sin(t1)
    
                if 0 <= x <= nx and 0 <= y <= ny:
                    corners.append((x, y))

    return corners
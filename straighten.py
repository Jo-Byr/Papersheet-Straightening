# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:55:32 2022

@author: jonat
"""

# Pb si l'image est en format paysage
# Pour le moment on fait du redressement puis on vérifie si on a besoin de rotation, à voir si on peut pas combiner les 2 par la suite
# Vérifier si les redressements horizontal et vertical sur une même image donnent un bon résultat

import cv2
import numpy as np
from math import ceil,floor

import matplotlib.pyplot as plt


def f(x0, x1, a, b):
    Y = np.asarray([b*(a+b-1)*x0, a*(a+b-1)*x1])
    Y *= 1/(b*(b-1)*x0 + a*(a-1)*x1 + a*b)
    return Y


def triangle_area(p0, p1, p2):
    a = np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    b = np.sqrt((p0[0] - p2[0])**2 + (p0[1] - p2[1])**2)
    c = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    S = (a+b+c)/2
    area = np.sqrt(S*(S-a)*(S-b)*(S-c))
    return area


def straighten(img):
    I = img.copy()

    # Grey-scaling the image
    if len(I.shape) == 3:
        G = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    else:
        G = I.copy()

    ny, nx = G.shape

    # Blurring to ignore the content of the paper
    blurred = cv2.GaussianBlur(G, (2*(nx//100)+1, 2*(nx//100)+1), 0)

    # Binarisation
    n = nx//30
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, n, n//4)

    # Hough transform
    lines = cv2.HoughLines(binary, 1, np.pi/180, nx//2)

    dr = nx//20  # 5% of nx margin for similarity
    dt = np.pi/18  # 10° margin for similarity

    found_lines = []
    i = 0

    # Searching for the 4 borders
    while len(found_lines) != 4 and i < len(lines):
        r, t = lines[i][0]
        treated = False
        for l in found_lines:
            if abs(r - l[0]) < dr and abs(t - l[1]) < dt:
                treated = True

        if not(treated):
            found_lines.append((r, t))

        i += 1

    # Visualisation
    for r, t in found_lines:
        a = np.cos(t)
        b = np.sin(t)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + ny*(-b))
        y1 = int(y0 + ny*(a))
        x2 = int(x0 - ny*(-b))
        y2 = int(y0 - ny*(a))

        cv2.line(I, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.figure()
    plt.imshow(I)

    # Reorganization
    P = []  # List of intersections points
    for i in range(3):
        for j in range(i+1, 4):
            r1 = found_lines[i][0]
            t1 = found_lines[i][1]
            r2 = found_lines[j][0]
            t2 = found_lines[j][1]
            x = np.tan(t1)*np.tan(t2)/(np.tan(t2)-np.tan(t1)) * \
                (r1/np.sin(t1) - r2/np.sin(t2))
            y = -x/np.tan(t1) + r1/np.sin(t1)

            if 0 <= x <= nx and 0 <= y <= ny:
                P.append((x, y))


    # Mapping Q points (corners of the target image) to P points
    Q = []
    
    size = np.sqrt(nx**2 + ny**2)
    for i in range(3):
        for j in range(i+1,4):
            dist = np.sqrt((P[i][0]-P[j][0])**2 + (P[i][1]-P[j][1])**2)
            if dist < size:
                size = dist
    dist = int(dist)
    sy,sx = int(1.414*dist),dist
    
    C = [(0, 0), (sx-1, 0), (0, sy-1), (sx-1, sy-1)]
    for x, y in P:
        dist = [np.sqrt(x**2 + y**2), np.sqrt((x-sx)**2 + y**2),
                np.sqrt(x**2 + (y-sy)**2), np.sqrt((x-sx)**2 + (y-sy)**2)]
        Q.append(C[dist.index(min(dist))])
    
    #Applying the transform
    A = np.asarray([[P[0][0] , P[0][1] , 1 , 0 , 0 , 0 , -Q[0][0]*P[0][0] , -Q[0][0]*P[0][1]],
                    [0 , 0 , 0 , P[0][0] , P[0][1] , 1 , -Q[0][1]*P[0][0] , -Q[0][1]*P[0][1]],
                    [P[1][0] , P[1][1] , 1 , 0 , 0 , 0 , -Q[1][0]*P[1][0] , -Q[1][0]*P[1][1]],
                    [0 , 0 , 0 , P[1][0] , P[1][1] , 1 , -Q[1][1]*P[1][0] , -Q[1][1]*P[1][1]],
                    [P[2][0] , P[2][1] , 1 , 0 , 0 , 0 , -Q[2][0]*P[2][0] , -Q[2][0]*P[2][1]],
                    [0 , 0 , 0 , P[2][0] , P[2][1] , 1 , -Q[2][1]*P[2][0] , -Q[2][1]*P[2][1]],
                    [P[3][0] , P[3][1] , 1 , 0 , 0 , 0 , -Q[3][0]*P[3][0] , -Q[3][0]*P[3][1]],
                    [0 , 0 , 0 , P[3][0] , P[3][1] , 1 , -Q[3][1]*P[3][0] , -Q[3][1]*P[3][1]]
                    ])
    
    UV = np.asarray([Q[0][0],Q[0][1],Q[1][0],Q[1][1],Q[2][0],Q[2][1],Q[3][0],Q[3][1]])
    UV = UV.reshape((8,1))
    M = np.dot(np.linalg.inv(A),UV)
    a,b,c,d,e,f,g,h = M
    
    
    if len(I.shape) == 3:
        res = np.zeros((sy,sx,3))
    else:
        res = np.zeros((sy,sx))
    res = res.astype(np.uint8)
    percent = 0
    for v in range(sy):
        if v/sy > percent/100:
            print(percent)
            percent += 1
        for u in range(sx):
            x = ((c-u)*(v*h-e) - (f-v)*(u*h-b))/((u*g-a)*(v*h-e) - (v*g-d)*(u*h-b))
            y = ((c-u)*(v*g-d) - (f-v)*(u*g-a))/((u*h-b)*(v*g-d) - (v*h-e)*(u*g-a))
            y,x = int(y),int(x)
            res[v,u] = img[y,x]
    
    return res


if __name__ == '__main__':
    I = cv2.imread('./Images/Rotation/im4.jpg')
    res = straighten(I)
    plt.figure()
    plt.imshow(res)
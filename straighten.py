# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:55:32 2022

@author: jonat
"""

#Pb si l'image est en format paysage
#Pour le moment on fait du redressement puis on vérifie si on a besoin de rotation, à voir si on peut pas combiner les 2 par la suite
#Vérifier si les redressements horizontal et vertical sur une même image donnent un bon résultat

import cv2
import numpy as np

import matplotlib.pyplot as plt

def straighten(img):
    I = img.copy()
    
    #Grey-scaling the image
    if len(I.shape) == 3:
        G = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
    else:
        G = I.copy()
        
    ny,nx = G.shape
    
    #Blurring to ignore the content of the paper
    blurred = cv2.GaussianBlur(G,(2*(nx//100)+1 , 2*(nx//100)+1) , 0)
    
    #Binarisation
    n = nx//30
    binary = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, n, n//5)
    
    #Hough transform
    lines = cv2.HoughLines(binary,1,np.pi/180,nx//2)
    
    dr = nx//20 #5% of nx margin for similarity
    dt = np.pi/18 #10° margin for similarity
    print(dt)
    
    found_lines = []
    i = 0
    
    #Searching for the 4 borders
    while len(found_lines) != 4 and i < len(lines):
        r,t = lines[i][0]
        treated = False
        for l in found_lines:
            if abs(r - l[0]) < dr and abs(t - l[1]) < dt:
                treated = True
                
        if not(treated):
            found_lines.append((r,t))
    
        i += 1
    
    #Visualisation
    for r,t in found_lines:
        a = np.cos(t)
        b = np.sin(t)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + ny*(-b))
        y1 = int(y0 + ny*(a))
        x2 = int(x0 - ny*(-b))
        y2 = int(y0 - ny*(a))
    
        cv2.line(I,(x1,y1),(x2,y2),(255,0,0),2)
    
    plt.figure()
    plt.imshow(I)
    
    #Reorganization
    temp = found_lines
    found_lines = [(np.sign(r)*t,abs(r)) for r,t in temp]
    
    #Identifying the case
    # found_lines.sort()
    # redressingV = False
    # redressingH = False
    # for i in range(3):
    #     for j in range(i+1,4):
    #         theta1 = found_line[i][0] - (found_line[i][0] - np.pi/2) * (found_line[i][0] > np.pi/2)
    #         theta2 = found_line[j][0] - (found_line[j][0] - np.pi/2) * (found_line[j][0] > np.pi/2)
    #         if (abs(theta1 - theta2) < np.pi/4) and (abs(found_line[i][0] - found_line[j][0]) > dt):
                
                
    # if abs(found_lines[0][0] - found_lines[1][0]) > dt:
    #     redressingH = True
    # if abs(found_lines[2][0] - found_lines[3][0]) > dt:
    #     redressingV = True
    
    # # if redressingH:
    # #     #Horizontal non-parallel lines
    # #     y1 = 
    
    # rotation = False
    # for i in range(4):
    #     theta = found_lines[i][0]%(np.pi/2)
    #     if abs(theta) > dt and abs(theta - np.pi/2) > dt:
    #         rotation = True
            
    # print(redressingH or redressingV,rotation)
    
    
    return found_lines

if __name__ == '__main__':
    for i in range(1,9):
        I = cv2.imread('./Images/Rotation/rot'+str(i)+'.jpg')
        L = straighten(I)
        #print(L)
    for i in range(1,5):
        I = cv2.imread('./Images/Redressement/red'+str(i)+'.jpg')
        L = straighten(I)
        #print(L)
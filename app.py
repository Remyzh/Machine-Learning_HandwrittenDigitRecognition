from tkinter.tix import DisplayStyle
from tokenize import triple_quoted
import pygame, sys
from pygame.locals import *       #import everything from .locals
import numpy as np
from keras.models import load_model
import cv2
WHITE = (255,255,255)  #numbers represent color channels
BLACK = (0,0,0)
RED = (255,0,0)
BOUNDRYINC = 5
WINDOWSIZEX = 640
WINDOWSIZEY = 480


IMAGESAVE = False

MODEL = load_model("bestmodel.h5")

#FONT = pygame.font.Font("freesansbold.tff", 18)

LABELS = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four",
        5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine" }

# Initialize our pygame
pygame.init()

FONT = pygame.font.SysFont("freesansbold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
WHITE_INT = DISPLAYSURF.map_rgb(WHITE)

iswriting = False
number_xcord = []
number_ycord = []
img_arr = []
PREDICT = True
img_count = 1

while True:
    for event in pygame.event.get():  #catches any event that happens with pygame
        if event.type ==QUIT:   #allows us to close window
            pygame.quit()
            sys.exit()
        
        #if mouse drew something, draw a cricle around it
        if event.type == MOUSEMOTION and iswriting:
            xcord,ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0 )

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        #stopped writing, now we resize the image, draw a rectangle around it, and run the model with image
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            print(number_ycord)

            rect_min_x = max(number_xcord[0] -BOUNDRYINC, 0)
            rect_max_x = min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            rect_min_Y = max(number_ycord[0] -BOUNDRYINC )
            rect_max_Y =  min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEX)

            number_xcord = []
            number_ycord = []

            ing_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x : rect_max_x, rect_min_Y, rect_max_Y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_count +=1
            
            if PREDICT:
                image = cv2.resize(img_arr, (28,28))
                image = np.pad(image, (10,10), 'constant', constant_values = 0)
                image = cv2.resize(image, (28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRec0bj = image.get_rect()           #testing? replaced with image
                textRec0bj.left,textRec0bj.bottom = rect_min_x, rect_max_Y 

                DISPLAYSURF.blit(textSurface, textRec0bj)
            
            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)

        pygame.display.update


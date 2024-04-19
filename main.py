from skimage.io import imread
from skimage.filters import gaussian
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float
import scipy as sp

def is_divisible_by_100(number):
    return number % 768 == 0

def model():

    a = np.array([
    [47.917, 0, -146.636, -141.572, -123.269],
    [0, 408.250, 68.487, 69.828, 53.479],
    [-146.636, 68.487, 2654.285, 2621.672, 2440.381],
    [-141.572, 69.828, 2621.672, 2597.818, 2435.368],
    [-123.269, 53.479, 2440.381, 2435.368, 2404.923]
    ])
    return a

def readImage(img):
    image = imread('target.jpg')
    return image

#display a given image
def showImage(image):
    plt.imshow(image)  
    plt.axis('off')  
    plt.show()


#this function collects every possible frame
def checkWindows(image):

    #set the height and width of the frames
    h = 70
    w = 24

    #initialize a way to store each frame and their location
    windows = []
    locations = []

    #find each possible frame by traversing the image one pixel at a time
    for i in range(0, image.shape[0], 1):
        for j in range(0, image.shape[1], 1):

            #append both the frame and its location to the relevent array
            windows.append(image[i:i+h, j:j+w])
            locations.append((i, j))

    return windows, locations

#this function finds the covariance of a given frame
def GetCovariance(window):
    #initialize a way to  store all pixels and locations in x  y r g b format
    dataHolder = np.zeros((5,window.shape[0]*window.shape[1]))

    #loop through every pixel in this window 
    for i in range(0, window.shape[0]):
        for j in range(0, window.shape[1]):
            
            #fill up array with pixels locations and RGB values
            dataHolder[:, (i*window.shape[1] + j)] = np.array([i, j, window[i, j, 0], window[i, j, 1], window[i, j, 2]])

    #return the covariance of all of these
    return np.cov(dataHolder, bias=True)


image = readImage('target.jpg')
model_matrix = model()
window_height = 70
window_width = 24
CurrentLocation = (0,0)
curBest = float("inf")

windows, locations = checkWindows(image)

for i in range(len(windows)):

    #display a percentage bar to track progress
    if is_divisible_by_100(i):
        percentage = (i / 76800) * 100
        formatted_percentage = f'{percentage:.2f}% COMPLETE'
        print(formatted_percentage)

    eigenW , trash = sp.linalg.eig(model(), GetCovariance(windows[i]))
    
    dist = np.sqrt((np.log(eigenW)**2).sum())
    
    if (dist < curBest) and (dist >= np.complex128(3.21+0j)):
        
        CurrentLocation = locations[i]
        curBest = dist
        
showImage(image[CurrentLocation[0]:CurrentLocation[0]+70, CurrentLocation[1]:CurrentLocation[1]+24])

from glob import glob
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os

class imageProcessingPipeline:
    
    def __init__(self,path_to_image_directory):
        recursiveStringForPath = path_to_image_directory+"/*/*"
        self.addresses = glob(recursiveStringForPath)  
        self.labelDict = {"Bedroom":0,"Bathroom":1, "Living":2,"Kitchen":3,"Game Room":4, "Lobby":5}
        self.imageToLabelMapping = {"Bedroom":[],"Bathroom":[], "Living":[],"Kitchen":[],"Game Room":[], "Lobby":[]}          

    def resizeImage(self,image,shape,interpolation = "bilinear"):
        if len(shape) == 3:
            width,height,_ = shape
        elif len(shape) == 2:
            width, height = shape
        else:
            print("Error: Shape must be a tuple of length 2 or 3!")
            return None
        if interpolation == "bilinear":
            resized_img = cv2.resize(image,(width,height),interpolation = cv2.INTER_LINEAR)
        elif interpolation == "bicubic":
            resized_img = cv2.resize(image,(width,height),interpolation = cv2.INTER_CUBIC)
        elif interpolation == "area":
            resized_img = cv2.resize(image,(width,height),interpolation = cv2.INTER_AREA)
        elif interpolation == "nearest":
            resized_img = cv2.resize(image,(width,height),interpolation = cv2.INTER_NEAREST)
        return resized_img   
    
    def cropBlackBorders(self,image):
        
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        cropped_image = image[y:y+h,x:x+w]
        return cropped_image

    def sharpenImage(self,image,sharpeningMatrix):
        
        if sharpeningMatrix is not None:
            return cv2.filter2D(image, -1, sharpeningMatrix)
        else:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(image, -1, kernel)
    
    def adjustBrightnessAndConstrast(self,image, alpha = 1., beta = 0.):
        """
        Function to adjust the brightness and contrast of the input image
        Inputs: 
        w x h x c image array
        alpha: parameter corresponding to the brightness adjustment
        beta: parameter corresponding to the contrast adjustment
        Returns:
        w x h x c image array with contrast and brightness adjusted as per 
        the input values.

        """
        new_image = np.zeros(image.shape, image.dtype)
        for w in range(image.shape[0]):
            for h in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[w,h,c] = np.clip(alpha*image[w,h,c] + beta, 0, 255)
        return new_image
    
    def generateLabelledData(self):
        
        start = time.time()
        print("Starting labelling of the images...")
        
        for eachfileAddress in self.addresses:
            if "Bedroom" in eachfileAddress:
                self.imageToLabelMapping["Bedroom"].append(cv2.imread(eachfileAddress))
            elif "Bathroom" in eachfileAddress:
                self.imageToLabelMapping["Bathroom"].append(cv2.imread(eachfileAddress))
            elif "Living" in eachfileAddress:
                self.imageToLabelMapping["Living"].append(cv2.imread(eachfileAddress))
            elif "Kitchen" in eachfileAddress:
                self.imageToLabelMapping["Kitchen"].append(cv2.imread(eachfileAddress))
            elif "Lobby" in eachfileAddress:
                self.imageToLabelMapping["Lobby"].append(cv2.imread(eachfileAddress))
            else:
                self.imageToLabelMapping["Game Room"].append(cv2.imread(eachfileAddress))

        end = time.time()
        print("Image-to-label mapping generated.")
        print(end - start,"secs")

    def writeImageToFolder(self,output_directory_path):
        
        start = time.time()
        print("Writing the images to the destination folder...")

        for category in self.labelDict.keys():
            dir = os.path.join(output_directory_path,category)
            if not os.path.exists(dir):
                os.mkdir(dir)
            totalImages,count = len(self.imageToLabelMapping[category]),1
            if totalImages != 0:
                images = self.imageToLabelMapping[category]
                for i in range(0,totalImages,1):
                    fileName = dir + "/image_"+ str(count)+".jpg"
                    cv2.imwrite(fileName,images[i])
                    count+=1
        end = time.time()
        print("Images saved.")
        print(end - start,"secs")
        
        
from glob import glob
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import Augmentor
import rawpy

class imageProcessingPipeline:
    
    def __init__(self,path_to_image_directory):
        """
        Inputs: String for the image directory path.

        Constructor for the imageProcessingPipeline class.
        Initializes the following class variables:

        1. addresses: list of all the image addresses obtained by recursive parsing.
        2. labelDict: dictionary mapping of string category label to int labels
        3. imageToLabelMapping: dictionary mapping of categories to their corresponding 
        list of images.

        Returns: None
        """
        recursiveStringForPath = path_to_image_directory+"/*/*"
        self.addresses = glob(recursiveStringForPath)  
        self.labelDict = {"Bedroom":0,"Bathroom":1, "Living":2,"Kitchen":3,"Game Room":4, "Lobby":5}
        self.imageToLabelMapping = {"Bedroom":[],"Bathroom":[], "Living":[],"Kitchen":[],"Game Room":[], "Lobby":[]}          

    def resizeImage(self,image,shape,interpolation = "bilinear"):
        """
        Inputs: 

        w x h x c image array
        (W,H): tuple of the new shape
        interpolation: method of interpolation, by default set to bilinear
        
        Function to resize the image dimensions.

        Returns:
        w x h x c image array adjusted as per the input values.
        """ 
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
            
    def rawToJPEGConverter(self, rawFilePath):
        """
        Inputs:
        inputFilePath: string path to raw file

        Function to convert .arw and .raw files to array format.

        Returns:
        image array w*h*c extracted from the raw file 
        """
        with rawpy.imread(rawFilePath) as raw:
            rgb = raw.postprocess(gamma = (4.0,4.5),output_bps=8)
        return np.asarray(rgb)
    
    def segregateDataByCategory(self):
        """
        Inputs: None

        For each file address, the function segregates the images by their corresponding labels
        which are present in the address string. 
        The results are stored in the imageToLabelMapping class variable.

        Returns: None
        """       
        start = time.time()
        print("Starting labelling of the images...")
        
        for eachfileAddress in self.addresses:
            if "Bedroom" in eachfileAddress:
                if ".ARW" in eachfileAddress or ".RAW" in eachfileAddress:
                    self.imageToLabelMapping["Bedroom"].append(self.rawToJPEGConverter(eachfileAddress))
                else:
                    self.imageToLabelMapping["Bedroom"].append(cv2.imread(eachfileAddress))
            elif "Bathroom" in eachfileAddress:
                if ".ARW" in eachfileAddress or ".RAW" in eachfileAddress:
                    self.imageToLabelMapping["Bathroom"].append(self.rawToJPEGConverter(eachfileAddress))
                else:
                    self.imageToLabelMapping["Bathroom"].append(cv2.imread(eachfileAddress))
            elif "Living" in eachfileAddress:
                if ".ARW" in eachfileAddress or ".RAW" in eachfileAddress:
                    self.imageToLabelMapping["Living"].append(self.rawToJPEGConverter(eachfileAddress))
                else:
                    self.imageToLabelMapping["Living"].append(cv2.imread(eachfileAddress))
            elif "Kitchen" in eachfileAddress:
                if ".ARW" in eachfileAddress or ".RAW" in eachfileAddress:
                    self.imageToLabelMapping["Kitchen"].append(self.rawToJPEGConverter(eachfileAddress))
                else:
                    self.imageToLabelMapping["Kitchen"].append(cv2.imread(eachfileAddress))
            elif "Lobby" in eachfileAddress:
                if ".ARW" in eachfileAddress or ".RAW" in eachfileAddress:
                    self.imageToLabelMapping["Lobby"].append(self.rawToJPEGConverter(eachfileAddress))
                else:
                    self.imageToLabelMapping["Lobby"].append(cv2.imread(eachfileAddress))
            else:
                if ".ARW" in eachfileAddress or ".RAW" in eachfileAddress:
                    self.imageToLabelMapping["Game Room"].append(self.rawToJPEGConverter(eachfileAddress))
                else:
                    self.imageToLabelMapping["Game Room"].append(cv2.imread(eachfileAddress))

        end = time.time()
        print("Image-to-label mapping generated.")
        print(end - start,"secs")

    def writeImageToFolder(self,output_directory_path):

        """
        Inputs: String input for output directory path

        Writes the edited/ modified images in the output directory path, with each image being stored
        in the subfolder corresponding to it's label.

        Returns: None
        """
        start = time.time()
        print("Writing the images to the destination folder...")

        for category in self.labelDict.keys():
            dir = os.path.join(output_directory_path,category)
            if not os.path.exists(dir):
                os.mkdir(dir)
            totalImages,count = len(self.imageToLabelMapping[category]),251
            if totalImages != 0:
                images = self.imageToLabelMapping[category]
                for i in range(0,totalImages,1):
                    fileName = dir + "/image_"+ str(count)+".jpg"
                    cv2.imwrite(fileName,images[i])
                    count+=1
        end = time.time()
        print("Images saved.")
        print(end - start,"secs")

    def generateLabelledData(self,category):
        """
        Inputs:
        category: String for the label which is to be assigned to each image.

        For each image the functin stores in the "images" list of the dict there is a 
        corresponding category label stored in the "labels" list of the dict.

        Return: Dict mapping with one image-one label combinations
        """
        labelPerImageDict = {"images":[],"labels":[]}
        labelPerImageDict["images"] = self.imageToLabelMapping[category]
        labelEncoded = self.labelDict[category]
        labelPerImageDict["labels"] = [labelEncoded for i in range(0,len(labelPerImageDict["images"]),1)]

        return labelPerImageDict
    
    def dataAugmentator(self,sampleCount,action = "write", output_directory_path = ""):
        """
        Inputs: 
        sampleCount: Number of output samples to be created.
        action: A variable that instructs the function on whether to store the 
        data generated or to return it as tuple of (image,labels) 
        output_directory_path: Path to the folder to house the augmented images 
        if action is "write" otherwise "" by default.

        For each image the functin stores in the "images" list of the dict there is a 
        corresponding category label stored in the "labels" list of the dict.

        Return: contingent on input to action variable, can be None or a tuple of (images,labels)
        """        
        totalImages, totalLabels = [],[]
        for category in self.labelDict.keys():
            inputs = self.generateLabelledData(category)
            images,labels =  inputs["images"],inputs["labels"]  
            if len(labels) == 0:
                continue
            else:
                for i in range(0,len(images),1):
                    # Add a <list> wrapper to each image for valid format as input to the Augmentor
                    images[i] = [images[i]] 
                augmentorObject = Augmentor.DataPipeline(images, labels)
                augmentorObject.rotate(1, max_left_rotation=5, max_right_rotation=5)
                augmentorObject.skew(probability = 0.5, magnitude=0.2)
                augmentorObject.flip_left_right(probability = 0.5)
                augmentorObject.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
                # augmentorObject.random_brightness(probability=0.4, min_factor=0.2,max_factor=0.5)
                augmented_images, labels = augmentorObject.sample(sampleCount)
                for i in range(0,len(augmented_images),1):
                    totalImages.append(augmented_images[i][0])
                    totalLabels.append(labels[i])
        if action == "write":
            # Map the images to their labels in the imagezToLabelMapping dictionary
            self.imageToLabelMapping = {"Bedroom":[],"Bathroom":[], "Living":[],"Kitchen":[],"Game Room":[], "Lobby":[]}  
            inv_map = {v: k for k, v in self.labelDict.items()}       
            for i in range(0,len(totalLabels),1):
                currKey = inv_map[totalLabels[i]]
            # The augmentor's list is nested so the images are encapsulated within another list within it
                self.imageToLabelMapping[currKey].append(totalImages[i])
            self.writeImageToFolder(output_directory_path)
        elif action == "return":
            return totalImages,totalLabels
        else:
            return None

        


            

        
        
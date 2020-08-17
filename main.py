import os
import sys
from imageProcessing import imageProcessingPipeline

def main():
    path_to_image_directory = sys.argv[0]
    output_directory_path = sys.argv[1]
    # path_to_image_directory = "C:/Users/gauri/Downloads/After Cleaning- JPG"
    # output_directory_path = "C:/Users/gauri/Downloads/After Cleaning- Preprocessed Images"
    # Initializing the image processing object
    preprocessingUnit = imageProcessingPipeline(path_to_image_directory)    
    # Reading the data and segregating them by labels
    preprocessingUnit.generateLabelledData()
    # Resizing images present with the object
    for currKey in preprocessingUnit.imageToLabelMapping.keys():
        currListOfImages = preprocessingUnit.imageToLabelMapping[currKey]
        print("Current set of images to be processed",len(currListOfImages))
        modifiedListOfImages = []
        for i in range(0,len(currListOfImages),1):
            originalImage = currListOfImages[i]
            resizedImage = preprocessingUnit.resizeImage(originalImage,(512,512),interpolation="bilinear")
            modifiedListOfImages.append(resizedImage)
        # replacing the original image with the new resized version
        preprocessingUnit.imageToLabelMapping[currKey] = modifiedListOfImages
        print("Total images processed",len(modifiedListOfImages))
    # Saving the preprocessed data to new folders in the output directory
    preprocessingUnit.writeImageToFolder(output_directory_path)

if __name__ == "__main__":
    # main function call
    main()
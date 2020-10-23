import os
import sys
from glob import glob
import numpy as np 
import random
import matplotlib.pyplot as plt #for testing the processed images
from imageProcessing import imageProcessingPipeline
from dataSampler import SamplingDataSet
from dataset import DataSet
from latentEncoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from kmeansAnalysis import kMeansAnalysis

def getData(path_to_image_directory,output_directory_path,imageSize,augCount):
    # Read the image
    preprocessingUnit = imageProcessingPipeline(path_to_image_directory)  
    preprocessingUnit.segregateDataByCategory()
    
    for key,listOfImages in preprocessingUnit.imageToLabelMapping.items():
        print("Current set of images to be processed",len(listOfImages),"belonging to category:",key)
        modifiedListOfImages = []
        for i in range(0,len(listOfImages),1):
            originalImage = listOfImages[i]
            tempImage = preprocessingUnit.resizeImage(originalImage,imageSize,interpolation="bicubic")
            modifiedListOfImages.append(tempImage)
        # replacing the original image with the new resized version
        preprocessingUnit.imageToLabelMapping[key] = modifiedListOfImages
    preprocessingUnit.dataAugmentator(augCount,output_directory_path=output_directory_path)   

def trainLatentEncoder(dataPath, numSamples, epochs, batchSize,learningRate, filters,latentDim):
    ## Sampling for training CAE
    addresses = glob(dataPath + "After Cleaning/*/*")
    addresses.extend(glob(dataPath + "Prior Cleaning/*/*"))
    samplesObj = SamplingDataSet(addresses,numSamples)
    samplesObj.extractSamples()
    images,labels = samplesObj.readSamples()
    # getting the data for training and validation
    getData = DataSet(images,labels,6,split_size = 0.15)
    X_train,X_test,_,_ = getData.train_test_split()
    # initialize the number of epochs to train for and batch size
    EPOCHS = epochs
    BS = batchSize
    w,h,c = X_train[0].shape
    # construct our convolutional autoencoder
    print("[INFO] building autoencoder...")
    (encoder, decoder, autoencoder) = ConvAutoencoder.build(w, h, c,filters=filters, latentDim=latentDim)
    opt = Adam(lr=learningRate,beta_1 = 0.1)
    autoencoder.compile(loss="mse", optimizer=opt)
    # train the convolutional autoencoder
    H = autoencoder.fit(
            X_train, X_train,
            validation_data = (X_test, X_test),
            epochs = EPOCHS,
            batch_size = BS,
            verbose = True,
            shuffle = True)
    return encoder, decoder, autoencoder

def runKMeansAnalysis(dataPath, sampleSize,encoderUnit):
    addresses = glob(dataPath + "After Cleaning/*/*")
    addresses.extend(glob(dataPath + "Prior Cleaning/*/*"))
    for size in sampleSize:
        samplesObj = SamplingDataSet(addresses,size)
        samplesObj.extractSamples()
        images,labels = samplesObj.readSamples()
        X_encoded = []
        for eachImage in images:
            eachImage = eachImage.astype("float32")
            eachImage = np.true_divide(eachImage,np.max(eachImage))
            img = np.expand_dims(eachImage,axis=0)
            X_encoded.append(encoderUnit.predict(img)[0])
        X_encoded = np.asarray(X_encoded)
        kmeansObj = kMeansAnalysis(X_encoded,labels,nClusters=6)
        model,y_kmeans = kmeansObj.gettingKMeansPredictions()
        kmeansObj.plotClusters3D(model,y_kmeans)

def main():
    # Read the image
    # path_to_image_directory = "C:/Users/gauri/Downloads/training data/After Cleaning-JPEG"
    # augmented_data_path = "C:/Users/gauri/Downloads/The Circular Hospitality Project- Data/After Cleaning"
    # getData(path_to_image_directory,augmented_data_path,(512,500),250)
    # --------------------------------------------------------------------------------------------------------------- #
    dataPath = "C:/Users/gauri/Downloads/The Circular Hospitality Project- Data/"
    encoder, _, _ = trainLatentEncoder(dataPath, 
                                                        numSamples = 10, 
                                                        epochs = 50,
                                                        batchSize = 8,
                                                        learningRate = 1e-4,
                                                        filters = (32,64),
                                                        latentDim = 64)
    # select a few random samples --> get their latent representation --> run kmeans on the images
    sampleSizes = [150]
    runKMeansAnalysis(dataPath=dataPath,sampleSize=sampleSizes,encoderUnit= encoder)

if __name__ == "__main__":
    # main function call
    main()
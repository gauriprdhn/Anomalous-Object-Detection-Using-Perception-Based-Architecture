import random
from glob import glob
from skimage import io
class SamplingDataSet:  
    def __init__(self,addresses,k=None,model_type = "clf"):
        self.addresses = addresses
        self.k = k
        self.model_type = model_type
        self.imagesByLabel = {"Bedroom":[],"Bathroom":[],"Living":[],"Kitchen":[],"Game Room":[],"Lobby":[]}
        self.labelDict = {"Bedroom":0,"Bathroom":1, "Living":2,"Kitchen":3,"Game Room":4, "Lobby":5}
        
    def checkCount(self,lst):
        return len(lst) == self.k
    
    def extractSamples(self):
        if self.k:
            for fileAddress in self.addresses:
                  random.shuffle(self.addresses)
                  if "Bedroom" in fileAddress:
                      if self.checkCount(self.imagesByLabel["Bedroom"]):
                          continue
                      else:
                          self.imagesByLabel["Bedroom"].append(fileAddress)
                  elif "Bathroom" in fileAddress:
                      if self.checkCount(self.imagesByLabel["Bathroom"]):
                          continue
                      else:
                          self.imagesByLabel["Bathroom"].append(fileAddress)
                  elif "Living" in fileAddress:
                      if self.checkCount(self.imagesByLabel["Living"]):
                          continue
                      else:
                          self.imagesByLabel["Living"].append(fileAddress)
                  elif "Kitchen" in fileAddress:
                      if self.checkCount(self.imagesByLabel["Kitchen"]):
                          continue
                      else:
                          self.imagesByLabel["Kitchen"].append(fileAddress)
                  elif "Game Room" in fileAddress:
                      if self.checkCount(self.imagesByLabel["Game Room"]):
                          continue
                      else:
                          self.imagesByLabel["Game Room"].append(fileAddress)
                  elif "Lobby" in fileAddress:
                      if self.checkCount(self.imagesByLabel["Lobby"]):
                          continue
                      else:
                          self.imagesByLabel["Lobby"].append(fileAddress)
        else:
                for fileAddress in self.addresses:
                    random.shuffle(self.addresses)
                    if "Bedroom" in fileAddress:
                        self.imagesByLabel["Bedroom"].append(fileAddress)
                    elif "Bathroom" in fileAddress:
                        self.imagesByLabel["Bathroom"].append(fileAddress)
                    elif "Living" in fileAddress:
                      self.imagesByLabel["Living"].append(fileAddress)
                    elif "Kitchen" in fileAddress:
                       self.imagesByLabel["Kitchen"].append(fileAddress)
                    elif "Game Room" in fileAddress:
                       self.imagesByLabel["Game Room"].append(fileAddress)
                    elif "Lobby" in fileAddress:
                        self.imagesByLabel["Lobby"].append(fileAddress)
        # apply check for count of samples per category
        self.testCountOfSamples()    

    def testCountOfSamples(self):
        if self.model_type == "clf":
            for key in self.imagesByLabel.keys():
                if len(self.imagesByLabel[key]) == 0:
                    raise  ValueError('The sampled list is empty! Check the addresses imported')
        elif self.model_type == "ae":
            count = 0
            for key in self.imagesByLabel.keys():
                if len(self.imagesByLabel[key]) == 0:
                  count += 1
            if count == len(self.imagesByLabel.keys()):
                  raise  ValueError('The sampled list is empty! Check the addresses imported')
        return None
        
    def readSamples(self):
        samples,labels = [],[]
        if self.testCountOfSamples():
            print("Error in reading the samples! One of the categories has no images")
        else:
            for key in self.imagesByLabel.keys():
                for img in self.imagesByLabel[key]:
                    image = io.imread(img)
                    samples.append(image)
                    if "Bathroom" in img:
                        labels.append(self.labelDict["Bathroom"])
                    elif "Bedroom" in img:
                        labels.append(self.labelDict["Bedroom"])
                    elif "Kitchen" in img:
                        labels.append(self.labelDict["Kitchen"])
                    elif "Living" in img:
                        labels.append(self.labelDict["Living"])
                    elif "Lobby" in img:
                        labels.append(self.labelDict["Lobby"])
                    elif "Game Room" in img:
                        labels.append(self.labelDict["Game Room"])

            return samples,labels
import numpy as np
import cv2
import pickle

class LoadData:
    '''
    Class to laod the data
    '''
    def __init__(self, data_dir, classes, h):
        '''
        :param data_dir: directory where the dataset is kept
        :param classes: number of classes in the dataset
        :param cached_data_file: location where cached file has to be stored
        :param normVal: normalization value, as defined in ERFNet paper
        '''
        self.data_dir = data_dir
        self.classes = classes
        self.h = h

    def compute_class_weights(self, histogram):
        '''
        Helper function to compute the class weights
        :param histogram: distribution of class samples
        :return: None, but updates the classWeights variable
        '''
        print(" original histogram " + str(histogram))
        normHist = histogram / np.sum(histogram)
        print(normHist)
        return normHist

    def readFile(self, fileName):
        '''
        Function to read the data
        :param fileName: file that stores the image locations
        :param trainStg: if processing training or validation data
        :return: 0 if successful
        '''

        global_hist = np.zeros(self.classes, dtype=np.float32)

        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
                line_arr = line.split(',')

                label_file = ((self.data_dir).strip() + '/' + line_arr[1].strip()).strip()
                label_img = cv2.imread(label_file, 0)
                label_img = cv2.resize(label_img, (self.h*2, self.h), interpolation=cv2.INTER_NEAREST)
                hist = np.histogram(label_img, self.classes)
                global_hist += hist[0]

            #compute the class imbalance information
            return self.compute_class_weights(global_hist)

    def processData(self):
        '''
        main_multiscale.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        '''
        print('Processing training data')
        return self.readFile('train.txt')


if __name__ == "__main__":

    ld =LoadData('D:\DATA\cityscape', 20, 256)
    print(ld.processData())
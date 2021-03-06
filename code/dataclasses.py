import numpy as np
import cv2
import os
import Utils 


class DATA():

    def __init__(self, dirname):
        self.dir_path = os.path.join(Utils.DATA_DIR, dirname)
        self.filelist = os.listdir(self.dir_path)
        self.batch_size = Utils.BATCH_SIZE
        self.size = len(self.filelist)
        self.data_index = 0

    def read_img(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        print(filename)
        labimg = cv2.cvtColor(cv2.resize(img, (Utils.IMAGE_SIZE, Utils.IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
        labimg_ori = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return np.reshape(labimg[:,:,0], (Utils.IMAGE_SIZE, Utils.IMAGE_SIZE, 1)), labimg[:, :, 1:], img, labimg_ori[:,:,0]



    def generate_batch(self):
        batch = []
        labels = []
        filelist = []
        labimg_oritList= []
        originalList = []
        for i in range(self.batch_size):
            filename = os.path.join(self.dir_path, self.filelist[self.data_index])
            filelist.append(self.filelist[self.data_index])
            greyimg, colorimg, original,labimg_ori = self.read_img(filename)
            batch.append(greyimg)
            labels.append(colorimg)
            originalList.append(original)
            labimg_oritList.append(labimg_ori)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)/255 # values between 0 and 1
        labels = np.asarray(labels)/255 # values between 0 and 1
        originalList = np.asarray(originalList)
        labimg_oritList = np.asarray(labimg_oritList)/255 # values between 0 and 1
        return batch, labels, filelist, originalList, labimg_oritList


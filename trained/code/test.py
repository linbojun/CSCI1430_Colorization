import numpy as np
import cv2
import os
import imageio
from tensorflow import keras
import Utils

class DATA():
    
    def __init__(self):
        self.dir_path = os.path.join("../val_data", "")
        self.test_data_path = os.path.join("../test_data", "")
        self.filelist = os.listdir(self.dir_path)
        self.batch_size = Utils.BATCH_SIZE
        self.size = len(self.filelist)
        self.data_index = 0

    def read_img(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        labimg = cv2.cvtColor(cv2.resize(img, (Utils.IMAGE_SIZE, Utils.IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
        labimg_ori = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return np.reshape(labimg[:,:,0], (Utils.IMAGE_SIZE, Utils.IMAGE_SIZE, 1)), labimg[:, :, 1:], img, labimg_ori[:,:,0]

    def generate_data(self):
        filelist = []
        self.input_images = []
        for i in range(self.size):
            filename = os.path.join(self.dir_path, self.filelist[self.data_index])
            filelist.append(self.filelist[self.data_index])
            greyimg, colorimg, original,labimg_ori = self.read_img(filename)
            self.data_index = (self.data_index + 1) % self.size

            #imageio.imwrite(Utils.OUT_DIR + "/" + str(self.data_index) + ".jpg", greyimg)
            self.input_images.append(greyimg)
        
        self.input_images = np.asarray(self.input_images)/255
        self.input_images_3 = np.tile(self.input_images, [1, 1, 1, 3])
        print ("data done !")
    
    def generate_result(self):
        self.generator = keras.models.load_model('../my_model_colorizationEpoch4.h5')

        #res = self.generator.predict(self.input_images_3)
        #print (res.shape)


if __name__ == "__main__":
    data = DATA()
    #data.generate_data()
    data.generate_result()

        

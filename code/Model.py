import tensorflow as tf
import numpy as np
import keras.layers as Layers
from Generator import Generator
from Discriminator import Discriminator
from Loss import wasserstein_loss, gradient_penalty_loss, GRADIENT_PENALTY_WEIGHT
import Utils
import os
from functools import partial
from keras import applications
from keras.callbacks import TensorBoard
from keras import backend
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

learning_rate = 0.00002

class Model():
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

        dis = Discriminator()
        self.discriminator = dis.get_model()
        self.discriminator.compile(loss=wasserstein_loss,
            optimizer=self.optimizer)

        gen = Generator()
        self.generator = gen.get_model()
        self.generator.compile(loss=['mse', 'kld'],
            optimizer=self.optimizer)

        img_l_3 = Layers.Input(shape=(Utils.IMAGE_SIZE, Utils.IMAGE_SIZE, 3))
        img_l = Layers.Input(shape=(Utils.IMAGE_SIZE, Utils.IMAGE_SIZE, 1))
        img_ab = Layers.Input(shape=(Utils.IMAGE_SIZE, Utils.IMAGE_SIZE, 2))

        self.generator.trainable = False
        predAB, classVector = self.generator(img_l_3)
        discPredAB = self.discriminator([predAB, img_l])
        discriminator_output_from_real_sample = self.discriminator([img_ab, img_l])
        
        averaged_samples = RandomWeightedAverage()._merge_function([img_ab, predAB])
        averaged_samples_out = self.discriminator([averaged_samples, img_l])
        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'


        self.discriminator_model = tf.keras.models.Model(inputs=[img_l, img_ab, img_l_3],
                            outputs=[discriminator_output_from_real_sample, discPredAB, averaged_samples_out])

        self.discriminator_model.compile(optimizer=self.optimizer,
                            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])


        self.generator.trainable = True
        self.discriminator.trainable = False
        self.combined = tf.keras.models.Model(inputs=[img_l_3, img_l],
                              outputs=[ predAB, classVector, discPredAB])
        self.combined.compile(loss=['mse','kld', wasserstein_loss], loss_weights=[1.0, 0.003, -0.1], optimizer=self.optimizer) #1/300


        self.log_path= os.path.join(Utils.LOG_DIR,Utils.TEST_NAME)
        self.callback = TensorBoard(self.log_path)
        self.callback.set_model(self.combined)
        self.train_names = ['loss', 'mse_loss', 'kullback_loss', 'wasserstein_loss']
        self.disc_names = ['disc_loss', 'disc_valid', 'disc_fake','disc_gp']


        self.test_loss_array = []
        self.g_loss_array = []

    def train(self, data,test_data, log,sample_interval=1):

        save_models_path =os.path.join(Utils.MODEL_DIR,Utils.TEST_NAME)
        if not os.path.exists(save_models_path):
                os.makedirs(save_models_path)

        VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)

        positive_y = np.ones((Utils.BATCH_SIZE, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((Utils.BATCH_SIZE, 1), dtype=np.float32)

        total_batch = int(data.size/Utils.BATCH_SIZE)

        for epoch in range(Utils.NUM_EPOCHS):
            for batch in range(total_batch):
                trainL, trainAB, _, original, l_img_oritList  = data.generate_batch()
                l_3=np.tile(trainL,[1,1,1,3])

                predictVGG =VGG_modelF.predict(l_3)

                g_loss =self.combined.train_on_batch([l_3, trainL],
                                                    [trainAB, predictVGG, positive_y])
                d_loss = self.discriminator_model.train_on_batch([trainL, trainAB, l_3], [positive_y, negative_y, dummy_y])

                #self.write_log(self.callback, self.train_names, g_loss, (epoch*total_batch+batch+1))
                #self.write_log(self.callback, self.disc_names, d_loss, (epoch*total_batch+batch+1))

                if (batch) % Utils.BATCH_SIZE == 0:
                    print("[Epoch %d] [Batch %d/%d] [generator loss: %08f] [discriminator loss: %08f]" %  ( epoch, batch,total_batch, g_loss[0], d_loss[0]))

            save_path = os.path.join(save_models_path, "my_model_combinedEpoch%d.h5" % epoch)
            self.combined.save_weights(save_path)
            save_path = os.path.join(save_models_path, "my_model_colorizationEpoch%d.h5" % epoch)
            self.generator.save_weights(save_path)
            save_path = os.path.join(save_models_path, "my_model_discriminatorEpoch%d.h5" % epoch)
            self.discriminator.save_weights(save_path)

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()


class RandomWeightedAverage():

    def _merge_function(self, inputs):
        weights = backend.random_uniform((Utils.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

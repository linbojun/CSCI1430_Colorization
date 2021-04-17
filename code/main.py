import os
from . import Utils
import datetime
import dataclasses
from .Model import Model

if __name__ == '__main__':
    
    # Create log folder if needed.
    log_path= os.path.join(Utils.LOG_DIR,Utils.TEST_NAME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, str(datetime.datetime.now().strftime("%Y%m%d")) + "_" + str(Utils.BATCH_SIZE) + "_" + str(Utils.NUM_EPOCHS) + ".txt"), "w") as log:
        log.write(str(datetime.datetime.now()) + "\n")

        print('load training data from '+ Utils.TRAIN_DIR)
        train_data = dataclasses.DATA(Utils.TRAIN_DIR)
        test_data = dataclasses.DATA(Utils.TEST_DIR)
        assert Utils.BATCH_SIZE<=train_data.size, "The batch size should be smaller or equal to the number of training images --> modify it in config.py"
        print("Train data loaded")

        print("Initiliazing Model...")
        colorizationModel = Model()
        print("Model Initialized!")

        print("Start training")
        colorizationModel.train(train_data,test_data, log)
        colorizationModel.save_weights('./checkpoints/my_colorization_checkpoints')

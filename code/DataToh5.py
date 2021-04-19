import numpy as np
import h5py
import os
import io
import sys
import argparse
import multiprocessing
import cv2
import Utils

image_size = Utils.IMAGE_SIZE
batch_size = Utils.BATCH_SIZE
num_cpus = multiprocessing.cpu_count()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data',
        default='../DATASET'+os.sep+'cifar100',
        help='Location where the dataset is stored.')   
    return parser.parse_args()

def process(f):
    global image_size
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return img

def main():
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    print("Successfully access to dataset")

    prefix = os.path.join(ARGS.data,"train")
    subdir_list = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))
    l = []
    for subdir_i in subdir_list:
        l += list(map(lambda x : os.path.join(subdir_i, x), os.listdir(subdir_i)))
    print("Successfully access to train dataset")

    i = 0
    imagenet = np.zeros((len(l), image_size, image_size, 3), dtype='uint8')
    pool = multiprocessing.Pool(num_cpus)
    while i < len(l):
        current_batch = l[i:i + batch_size]  
        current_res = np.array(pool.map(process, current_batch))
        imagenet[i:i + batch_size] = current_res    
        i += batch_size
        print(i, 'images')
    print("Successfully process train dataset")

    # Test
    prefix = os.path.join(ARGS.data,"test")
    l_test = list(map(lambda x : os.path.join(subdir_i, x), os.listdir(subdir_i)))
    print("Successfully access to val dataset")

    i = 0
    imagenet_test = np.zeros((len(l_test), image_size, image_size, 3), dtype='uint8')
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    while i < len(l_test):
        current_batch = l_test[i:i + batch_size]    
        current_res = np.array(pool.map(process, current_batch))
        imagenet_test[i:i + batch_size] = current_res    
        i += batch_size
        print(i, 'images')
    print("Successfully process test dataset")

    # Val
    prefix = os.path.join(ARGS.data,"val")
    subdir_list = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))
    l_val = []
    for subdir_i in subdir_list:
        l_val += list(map(lambda x : os.path.join(subdir_i, x), os.listdir(subdir_i)))
    print("Successfully access to val dataset")

    i = 0
    imagenet_val = np.zeros((len(l_val), image_size, image_size, 3), dtype='uint8')
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    while i < len(l_val):
        current_batch = l_val[i:i + batch_size]    
        current_res = np.array(pool.map(process, current_batch))
        imagenet_val[i:i + batch_size] = current_res    
        i += batch_size
        print(i, 'images')
    print("Successfully process val dataset")

    output_name = Utils.DATA_DIR +os.sep+ os.path.basename(ARGS.data) + ".h5py"
    with h5py.File(output_name, 'w') as f:
        f['train'] = imagenet
        f['test'] = imagenet_test
        f['val'] = imagenet_val
    print("Successfully done")

ARGS = parse_args()
main()
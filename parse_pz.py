import os
import gzip
import pickle
import argparse

import cv2
import numpy as np

from deepgtav.messages import frame2numpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse .pz files')
    parser.add_argument('-d', '--dataset_path', default='checkpoints', help='Place to store the dataset')
    parser.add_argument('--file_prefix', required=True, help='File to be parsed')
    parser.add_argument('--show', action='store_true', help='Display while parsing')
    args = parser.parse_args()

    save_folder = os.path.join(args.dataset_path, args.file_prefix)
    img_folder = os.path.join(save_folder, 'img')
    file_name = args.file_prefix + '.pz'

    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    if os.path.exists(os.path.join(args.dataset_path, file_name)):
        os.rename(os.path.join(args.dataset_path, file_name), os.path.join(save_folder, file_name))

    commands = []
    with gzip.open(os.path.join(save_folder, file_name), 'rb') as f:
        while True:
            try:
                message = pickle.load(f)

                idx = len(commands)
                commands.append([idx, message['steering'], message['throttle'], message['brake']])
                frame = frame2numpy(message['frame'], (320,160))

                cv2.imwrite(os.path.join(img_folder, '%05d.png' % idx), frame)

                if args.show:
                    cv2.imshow('GTAV---{}'.format(args.file_prefix), frame)
                    cv2.waitKey(10)
            except:
                break

        np.savetxt(os.path.join(save_folder, 'commands.csv'), np.asarray(commands), delimiter=',')
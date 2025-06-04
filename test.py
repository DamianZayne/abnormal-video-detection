from utils.util import psnr_error, load, load_frame
# import tensorflow._api.v2.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()
# tf.disable_v2_behavior()
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os
from constant import const
from models import prediction_networks_dict
import base64
import numpy as np
from flask import Flask, render_template, request, Response, send_file, url_for, redirect
# import redis
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img

import pathlib

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPUS[0]
dataset_name = const.DATASET
train_folder = const.TRAIN_FOLDER
test_folder = const.TEST_FOLDER
frame_mask = const.FRAME_MASK
pixel_mask = const.PIXEL_MASK
k_folds = const.K_FOLDS
kth = const.KTH
interval = const.INTERVAL
IMAGE_DTYPE = "float32"

IMAGE_QUEUE = "image_queue"

batch_size = const.BATCH_SIZE
iterations = const.ITERATIONS
num_his = const.NUM_HIS
height, width = const.HEIGHT, const.WIDTH

prednet = prediction_networks_dict[const.PREDNET]
evaluate_name = const.EVALUATE

margin = const.MARGIN
lam = const.LAMBDA

summary_dir = const.SUMMARY_DIR
snapshot_dir = const.SNAPSHOT_DIR
psnr_dir = const.PSNR_DIR
print(const)

folder = 'data/my'
video_name = '01'
data_path = pathlib.Path(folder)
# 将8张图片加载到内存中，并按顺序排列
print(data_path)


def base64_encode_image(img):
    return base64.b64encode(img).decode("utf-8")


def base64_decode_image(img, dtype):
    img = np.frombuffer(base64.decodebytes(img), dtype=dtype)
    return img


# 模型
def model_t(processed_batches, gt):
    train_anchor_output, train_anchor_feature, _ = prednet(processed_batches, use_decoder=True)
    print(train_anchor_output)
    print(gt)
    psnr_tensor = psnr_error(train_anchor_output, gt)
    print(psnr_tensor)
    return train_anchor_output, psnr_tensor


print(data_path)

tf.config.run_functions_eagerly(True)

config = tf.ConfigProto()


# db = redis.StrictRedis(host="localhost", port=6379, db=0)
def load_model(folder):
    pnsr = 0
    data_path = pathlib.Path(folder)
    # 将8张图片加载到内存中，并按顺序排列
    print(data_path)
    image_paths = []
    for i in range(0, 4):
        path = '000' + str(i) + '.jpg'
        image_paths.append(str(data_path) + '/' + str(path))
        print(image_paths[i])

    tf.config.run_functions_eagerly(True)
    video_clip = []
    for filename in image_paths:
        video_clip.append(load_frame(filename, height, width))
    num_batches = len(video_clip) // batch_size
    image_batches = np.array_split(video_clip, num_batches)  # 4 2 224 224 3
    video_clip = np.stack(image_batches, axis=0)
    video_clip = tf.stack(video_clip, axis=0)
    gt = video_clip[:, -1, ...]
    processed_batches = video_clip[:, 0:4, ...]
    config = tf.ConfigProto()
    with tf.variable_scope('generator', reuse=None):
        print(processed_batches)
        train_anchor_output, train_anchor_feature, _ = prednet(processed_batches, use_decoder=True)
        print("out{}", format(train_anchor_output))
        print(gt)
        psnr_tensor = psnr_error(train_anchor_output, gt)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('Init successfully!')
        restore_var = [v for v in tf.global_variables()]
        loader = tf.train.Saver(var_list=restore_var)

        def inference_func(ckpt, dataset_name, evaluate_name):
            loader.restore(sess, ckpt)
            print("Restored model parameters from {}".format(snapshot_dir))
            result = sess.run(train_anchor_output)
            p_e = sess.run(psnr_tensor)
            psnr = p_e
            print(p_e)
            gt_o = sess.run(gt)
            print("p_e")
            # predicted_image = array_to_img(result[0])
            predicted_image = array_to_img(result[0])
            print("save sucessfully")
            predicted_image.save('.data/result/img_4.jpg')
            # plt.imshow(predicted_image)
            # plt.axis('off')  # 可选：关闭坐标轴
            # plt.show()

        def check_ckpt_valid(ckpt_name):
            is_valid = False
            ckpt = ''
            if ckpt_name.startswith('model.ckpt-'):
                ckpt_name_splits = ckpt_name.split('.')
                ckpt = str(ckpt_name_splits[0]) + '.' + str(ckpt_name_splits[1])
                ckpt_path = os.path.join(snapshot_dir, ckpt)
                if os.path.exists(ckpt_path + '.index') and os.path.exists(ckpt_path + '.meta') and \
                        os.path.exists(ckpt_path + '.data-00000-of-00001'):
                    is_valid = True

            return is_valid, ckpt

        def scan_psnr_folder():
            tested_ckpt_in_psnr_sets = set()
            for test_psnr in os.listdir(psnr_dir):
                tested_ckpt_in_psnr_sets.add(test_psnr)
            return tested_ckpt_in_psnr_sets

        def scan_model_folder():
            saved_models = set()
            for ckpt_name in os.listdir(snapshot_dir):
                is_valid, ckpt = check_ckpt_valid(ckpt_name)
                if is_valid:
                    saved_models.add(ckpt)
            return saved_models

        tested_ckpt_sets = scan_psnr_folder()
        print(tested_ckpt_sets)

        all_model_ckpts = scan_model_folder()
        print(all_model_ckpts)
        new_model_ckpts = all_model_ckpts - tested_ckpt_sets
        print(new_model_ckpts)

        for ckpt_name in new_model_ckpts:
            print("ddddddd")
            # inference
            ckpt = os.path.join(snapshot_dir, ckpt_name)
            print("-------------------------------")
            inference_func(ckpt, dataset_name, evaluate_name)

            tested_ckpt_sets.add(ckpt_name)

    return pnsr


if __name__ == '__main__':
    folder = "./data/myimage/"
    folder2 = "./static/"
    psnr = load_model(folder)
    out_p = ""
    print(psnr)
'''
python test.py  --dataset  avenue              --prednet  cyclegan_convlstm              --num_his  4                               --label_level  temporal                      --gpu      0                               --summary_dir  ./outputs3/summary/temporal/avenue/prednet_cyclegan_convlstm_folds_10_kth_1_/MARGIN_1.0_LAMBDA_1.0            --psnr_dir ./outputs/psnrs/temporal/avenue/prednet_cyclegan_convlstm_folds_10_kth_1_/MARGIN_1.0_LAMBDA_1.0           --interpolation  --snapshot_dir  ./outputs/checkpoints/temporal/avenue/prednet_cyclegan_convlstm_folds_10_kth_1_/MARGIN_1.0_LAMBDA_1.0/model.ckpt-18000/
'''
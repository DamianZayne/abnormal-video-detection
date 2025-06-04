from utils.util import psnr_error, load,load_frame
import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
import os
from constant import const
from models import prediction_networks_dict
import base64
import io
import uuid
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, Response, send_file
import redis
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array,array_to_img
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


folder = 'data/avenue/testing/frames/01'
video_name = '01'
data_path = pathlib.Path(folder)
# 将8张图片加载到内存中，并按顺序排列
print(data_path)



def base64_encode_image(img):
    return base64.b64encode(img).decode("utf-8")


def base64_decode_image(img, dtype):
    img = np.frombuffer(base64.decodebytes(img), dtype=dtype)
    return img

#模型
def model_t(processed_batches,gt):
    train_anchor_output, train_anchor_feature, _ = prednet(processed_batches, use_decoder=True)
    print(train_anchor_output)
    print(gt)
    psnr_tensor = psnr_error(train_anchor_output, gt)
    return train_anchor_output ,psnr_tensor

# sess = tf.Session()
print(data_path)

tf.config.run_functions_eagerly(True)

config = tf.ConfigProto()

db = redis.StrictRedis(host="localhost", port=6379, db=0)

app = Flask(__name__)
@app.route('/up',methods=['GET','POST'])
def up():
    if request.method=='GET':
        return render_template('up.html')
    elif request.method=='POST':
        files = request.files.getlist("myimg")
        folder="./data/myimage/"
        i=0
        # button_value = request.form['button']
        for objfile in files:
            obj_name=objfile.filename
            strpath =  folder+ '000' + str(i) + '.jpg'
            if not os.path.isdir(folder):
                os.makedirs("./data/myimage/")
            objfile.save(strpath)
            i=i+1

        data_path = pathlib.Path(folder)
        # 将8张图片加载到内存中，并按顺序排列
        print(data_path)
        image_paths = []
        for i in range(0, 4):
            path = '000' + str(i) + '.jpg'
            image_paths.append(str(data_path) + '/' + str(path))
            print(image_paths[i])

        tf.config.run_functions_eagerly(True)
        # im=read_video_clip(image_batches)
        video_clip = []
        for filename in image_paths:
            video_clip.append(load_frame(filename, height, width))
        num_batches = len(video_clip) // batch_size
        image_batches = np.array_split(video_clip, num_batches)  # 4 2 224 224 3
        video_clip = np.stack(image_batches, axis=0)
        video_clip = tf.stack(video_clip, axis=0)
        # processed_batches = []
        gt = video_clip[:, -1, ...]
        processed_batches = video_clip[:, 0:4, ...]
        config = tf.ConfigProto()
        with tf.variable_scope('generator', reuse=None):
            print(processed_batches)
            train_anchor_output, train_anchor_feature, _ = prednet(processed_batches, use_decoder=True)
            print(train_anchor_output)
            print(gt)
            psnr_tensor = psnr_error(train_anchor_output, gt)

        # sess = tf.Session()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            print('Init successfully!')
            restore_var = [v for v in tf.global_variables()]
            loader = tf.train.Saver(var_list=restore_var)

            def inference_func(ckpt, dataset_name, evaluate_name):
                loader.restore(sess, ckpt)
                print("Restored model parameters from {}".format(snapshot_dir))

                # load(loader, sess, snapshot_dir)

                # [0.28627455  0.13725495  0.15294123]
                # [0.6862745   0.5372549   0.54509807]
                # [0.6784314   0.5294118   0.54509807]]]]]
                result = sess.run(train_anchor_output)
                p_e = sess.run(psnr_tensor)
                gt_o = sess.run(gt)
                # print(result)
                # print(gt_o)
                print(p_e)
                predicted_image = array_to_img(result[0])
                predicted_image.save('./data/result/1.jpg')
                plt.imshow(predicted_image)
                plt.axis('off')  # 可选：关闭坐标轴
                plt.show()

            # if os.path.isdir(snapshot_dir):  # 于扫描指定目录下的文件，并将文件名存储在一个集合中。

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

            all_model_ckpts = scan_model_folder()
            new_model_ckpts = all_model_ckpts - tested_ckpt_sets

            for ckpt_name in new_model_ckpts:
                # inference
                ckpt = os.path.join(snapshot_dir, ckpt_name)
                inference_func(ckpt, dataset_name, evaluate_name)

                tested_ckpt_sets.add(ckpt_name)

        return render_template('result.html')
            # 创建内存缓冲区
        # buffer = io.BytesIO()

        # 将图像数据写入缓冲区
        # image_dat="./data/result/1.jpg"
        # image = Image.open(image_dat).tobytes()
        # buffer.write(image)
        #
        # # 将缓冲区指针移动到起始位置
        # buffer.seek(0)
        #
        # # 使用 Flask 的 send_file 函数将图像字节流发送给客户端
        # return send_file(buffer, mimetype='image/jpeg')


if __name__ =='__main__':
    app.run()
    # video_clips_tensor = tf.placeholder(shape=[1, (num_his + 1), height, width, 3], dtype=tf.float32)  # 1 4 256 256 3
    # inputs = video_clips_tensor[:, 0:num_his, ...]
    # pre = cyclegan_convlstm(inputs=inputs)



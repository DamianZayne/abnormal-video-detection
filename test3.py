import uuid

from load import load_model
from utils.util import psnr_error, load,load_frame
import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
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
import redis
from gevent import pywsgi
import pathlib
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
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

# r = redis.StrictRedis(host="127.0.0.1", port=6379, db=0,decode_responses=True)
from werkzeug.security import generate_password_hash, check_password_hash

folder = 'data/myimage'
video_name = '01'
data_path = pathlib.Path(folder)
# 将8张图片加载到内存中，并按顺序排列
print(data_path)
USERS = [
    {
        "id":1,
        "name":"admin",
        "password": generate_password_hash('123')
    },
    {
        "id":2,
        "name":"admin2",
        "password": generate_password_hash('123')
    }

]
def create_user(username, password):
    # 创建用户
    user={
        "id": uuid.uuid4(),
        "name":username,
        "password": generate_password_hash(password)
    }
    USERS.append(user)

def get_user(username):
    for user in USERS:
        if user.get("name")==username:
            return user
    return None





#模型


print(data_path)

tf.config.run_functions_eagerly(True)

config = tf.ConfigProto()

# db = redis.StrictRedis(host="localhost", port=6379, db=0)

# 假设的用户名和密码
USERNAME = 'admin'
PASSWORD = 'password'


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# 配置 Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # 未登录时跳转到登录页
class User(UserMixin):
    def __init__(self, user):
        self.id =user.get("id")
        self.name = user.get("name")
        self.password = user.get("password")

    def veryify_password(self, password):
        #密码验证
        return check_password_hash(self.password, password)

    def get_id(self):
        #获取id
        return self.id

    def get(user_id):
        """根据用户ID获取用户实体，为 login_user 方法提供支持"""
        if not user_id:
            return None
        for user in USERS:
            if user.get('id') == user_id:
                return User(user)
        return None




@app.route('/')
def index():
    return render_template('login.html')
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/login', methods=['Get','POST'])
def login():
    if request.method=='GET':
        return render_template('login.html')
    else:
        username = request.form['username']
        password = request.form['password']
        print(password)
        user_info=get_user(username)
        if user_info is None:
            return render_template('login.html', error='账号或者密码错误')
        else:
            user= User(user_info)
            if user.veryify_password(password):
                login_user(user)
                next_page = request.args.get('next')
                return redirect(url_for('up') or next_page)
            else:
                return render_template('login.html', error='账号或者密码错误')

@app.route('/logout',methods=['POST'])#登出
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/up',methods=['GET','POST'])
@login_required  # Flask-Login 提供的装饰器
def up():
    if request.method=='GET':
        return render_template('up.html')
    elif request.method=='POST':

        folder = "./data/myimage/"
        folder2 = "./static/"
        if 'get_up' in request.form:
            files = request.files.getlist("myimg")
            print(files)
            i=0
            # button_value = request.form['button']
            for objfile in files:
                obj_name=objfile.filename
                strpath =  folder+ '000' + str(i) + '.jpg'
                strpath2 =  folder2+ '000' + str(i) + '.jpg'
                # r.set(name=i,value= strpath)
                # r.set(name=i,value= strpath2)
                if not os.path.isdir(folder):
                    os.makedirs("./data/myimage/")
                objfile.save(strpath)
                objfile.seek(0)
                objfile.save(strpath2)
                i=i+1

            return redirect(url_for('up'))

        elif 'show' in request.form:
            return render_template('show.html')

        elif 'test' in request.form:
            psnr=load_model(folder)
            out_p=""
            if psnr >=20 :
                out_p="正常"
            elif psnr<20:
                out_p="异常"
            return render_template('result.html',psnr=psnr,out_p=out_p)


if __name__ =='__main__':
    # server = pywsgi.WSGIServer(('0.0.0.0', 5555), app)
    # server.serve_forever()
    # app.run(port=5555,debug=True)
    app.run()
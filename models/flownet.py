import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

from flownet2.src.flownet_sd.flownet_sd import FlowNetSD
from flownet2.src.net import Mode
from flownet2.src.training_schedules import LONG_SCHEDULE


def flownet(input_a, input_b, height, width, reuse=None):
    net = FlowNetSD(mode=Mode.TEST)
    # train preds flow
    input_a = (input_a + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    input_b = (input_b + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    # input size is 384 x 512
    input_a = tf.image.resize_images(input_a, [height, width])
    input_b = tf.image.resize_images(input_b, [height, width])
    flows = net.model(
        inputs={'input_a': input_a, 'input_b': input_b},
        training_schedule=LONG_SCHEDULE,
        trainable=False, reuse=reuse
    )
    return flows['flow']

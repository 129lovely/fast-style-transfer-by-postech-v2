from PIL import Image, ImageFont, ImageDraw, ImageTk
from datetime import date, datetime

import numpy as np
import tensorflow as tf

import cv2
import transform
import ctypes


class StyleTransfer:
    def __init__(self, height, width, models):
        self.idx = 0  # index of model
        self.sess = self.get_session()  # tensorflow session
        self.img_shape = (height, width, 3)
        self.batch_shape = (1,) + self.img_shape  # add one dim for batch
        self.models = models
        self.config_graph()
        self.load_checkpoint()  # load checkpoint from first model

    # open tensorflow session
    def get_session(self):
        g = tf.Graph()
        soft_config = tf.ConfigProto(allow_soft_placement=True)
        soft_config.gpu_options.allow_growth = True
        return tf.Session(config=soft_config)

    # load pre-trained model
    def load_checkpoint(self):
        saver = tf.train.Saver()
        model_path = self.models[self.idx]["path"]
        try:
            saver.restore(self.sess, model_path)
            print("load checkpoint: ", model_path)
            return True
        except:
            print("checkpoint %s not loaded correctly" % model_path)
            return False

    # config graph
    def config_graph(self):
        # graph input
        self.x = tf.placeholder(tf.float32, shape=self.img_shape, name="input")
        x_batch = tf.expand_dims(self.x, 0)  # add one dim for batch

        # result image from transform
        transform_instance = transform.Transform()
        self.preds = transform_instance.net(x_batch / 255.0)
        self.preds = tf.squeeze(self.preds)  # remove one dim for batch
        self.preds = tf.clip_by_value(self.preds, 0.0, 255.0)

    # get style image
    def get_style(self):
        return cv2.imread(self.models[self.idx]["img"])

    # get style transfered image
    def get_output(self, frame):
        output = self.sess.run(self.preds, feed_dict={self.x: frame})
        output = output[:, :, [2, 1, 0]].reshape(self.img_shape)
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        output = cv2.resize(output, (frame.shape[1], frame.shape[0]))
        return output

    # change style
    def change_style(self, is_prev=True):
        if is_prev:
            self.idx = (self.idx + len(self.models) - 1) % len(self.models)
        else:
            self.idx = (self.idx + len(self.models) + 1) % len(self.models)
        self.load_checkpoint()

    # get title and artist of style
    def get_style_info(self):
        return "「" + self.models[self.idx]["title"] + "」" + ", " + self.models[self.idx]["artist"]


class Cam:
    def __init__(self, device_id, inp_width, disp_width, disp_height):
        self.cam = cv2.VideoCapture(device_id)
        self.set_frame_shape(inp_width, disp_width, disp_height)

    # set frame shape
    def set_frame_shape(self, inp_width, disp_width, disp_height):
        """ get cam shape """
        cam_width, cam_height = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH), self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("cam width: %f, cam height: %f" % (cam_width, cam_height))

        """ set input frame shape for transform """
        self.inp_width = inp_width if inp_width % 4 == 0 else inp_width + 4 - (inp_width % 4)  # must be divisible by 4
        inp_height = int(inp_width * float(cam_height / cam_width))  # keep aspect ratio
        self.inp_height = (
            inp_height if inp_height % 4 == 0 else inp_height + 4 - (inp_height % 4)
        )  # must be divisible by 4

        """ set label frame shape """
        out_width = int(disp_height * float(self.inp_width / self.inp_height))
        self.lab_width = disp_width - out_width
        self.lab_height = int(self.lab_width * float(cam_height / cam_width))  # keep aspect ratio

    # set input and label frame
    def set_frame(self):
        ret, frame = self.cam.read()

        if not ret:
            print("could not receive frame")
            self.cam.release()
            return

        frame = cv2.flip(frame, 1)

        self.inp_frame = cv2.resize(frame, (self.inp_width, self.inp_height))
        self.lab_frame = cv2.resize(frame, (self.lab_width, self.lab_height))

    def get_frame(self):
        return self.inp_frame, self.lab_frame

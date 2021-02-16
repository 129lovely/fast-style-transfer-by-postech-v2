import cv2
import numpy as np
import tensorflow as tf
import transform


class Theme:
    def __init__(self, sess, height, width, models):
        self.idx = 0  # index of model
        self.sess = sess
        self.img_shape = (height, width, 3)
        self.batch_shape = (1,) + self.img_shape  # add one dim for batch
        self.models = models
        self.config_graph()
        self.load_checkpoint()  # load checkpoint from first model

    # load pre-trained model
    def load_checkpoint(self):
        saver = tf.train.Saver()
        model_path = self.models[self.idx]["path"]
        print(model_path)
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


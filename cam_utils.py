# save output image
with open("filename.jpg", "wb") as file:
    Image.fromarray(output).save(file, "jpeg")

from numpy as np
import cv2

class Cam:
    def __init__(self, models, sess, x, preds, width, height):
        self.models = 
        self.sess = sess
        self.x = x
        self.preds = preds
        self.width = width
        self.height = height

    # use a different syntax to get video size in OpenCV 1~2 and OpenCV 3~4
    def get_camera_shape(cam):
        cv_version_major, _, _ = cv2.__version__.split(".")
        print("cv_version_major: ", cv_version_major)
        if cv_version_major == "3" or cv_version_major == "4":
            return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    # load pre-trained model
    def load_checkpoint(idx_model):
        saver = tf.train.Saver()
        model_path = self.models[idx_model]["path"]
        try:
            saver.restore(self.sess, model_path)
            print("load checkpoint: ", model_path)
            return True
        except:
            print("checkpoint %s not loaded correctly" % model_path)
            return False

    def get_output(image):
        output = self.sess.run(self.preds, feed_dict={self.x: image})
        output = output[:, :, [2, 1, 0]].reshape(self.x.shape)
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        output = cv2.resize(output, (self.width, self.height))

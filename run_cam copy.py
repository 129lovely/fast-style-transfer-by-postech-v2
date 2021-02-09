# import packages
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import numpy as np
import cv2
import transform, my_utils  # import transform.py, cam_utils.py

import os, pdb, argparse
from datetime import date, datetime

### define variables
date_obj = date.today()
year = date_obj.year
month = date_obj.month
day = date_obj.day


"""
path: file path where model checkpoint is
img: file path where style image is
artist: artist's name
"""
models = [
    {"path": "models/model_eunsook_batch_8/final.ckpt", "img": "style/eunsook.jpg", "artist": "Park Eun Sook",},
    {"path": "models/model_gohg/final.ckpt", "img": "style/gohg.jpg", "artist": "gohg",},
    {"path": "models/model_unknown02/final.ckpt", "img": "style/unknown02.jpg", "artist": "gohg",},
    {"path": "models/model_wood/final.ckpt", "img": "style/wood.jpg", "artist": "gohg",},
]


"""
device_id: order of camera device
width: width of camera display
disp_width: width of entire display
horizontal: is display wide horizontally
num_sec: changing style interval
"""
opts = {
    "device_id": 0,
    "width": 700,
    "disp_width": 1250,
    "disp_source": 1,
    "horizontal": 1,
    "num_sec": 10,
}


# use a different syntax to get video size in OpenCV 1~2 and OpenCV 3~4
def get_camera_shape(cam):
    cv_version_major, _, _ = cv2.__version__.split(".")
    print("cv_version_major: ", cv_version_major)
    if cv_version_major == "3" or cv_version_major == "4":
        return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


# make full frame
def make_triptych(disp_width, frame, style, output, horizontal=True):
    print("make triptych")
    ch, cw, _ = frame.shape  # cam shape
    oh, ow, _ = output.shape  # output display shape

    disp_height = int(disp_width * (oh / ow))
    h = int((ch / cw) * disp_width * 0.5)
    w = int((cw / ch) * disp_height * 0.5)

    if horizontal:
        full_img = np.concatenate(
            [cv2.resize(frame, (int(w), int(0.5 * disp_height))), cv2.resize(style, (int(w), int(0.5 * disp_height))),],
            axis=0,
        )
        full_img = np.concatenate([full_img, cv2.resize(output, (disp_width, disp_height))], axis=1)
    else:
        full_img = np.concatenate(
            [cv2.resize(frame, (int(0.5 * disp_width), h)), cv2.resize(style, (int(0.5 * disp_width), h))], axis=1
        )
        full_img = np.concatenate([full_img, cv2.resize(output, (disp_width, disp_width * oh // ow))], axis=0)
    return full_img


def main(device_id, width, disp_width, disp_source, horizontal, num_sec):
    # define variables
    t1 = datetime.now()

    # config tensorflow options
    device_t = "/gpu:0"
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    # config font
    fontpath = "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf"
    font = ImageFont.truetype(font=fontpath, size=150)

    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
        cam = cv2.VideoCapture(device_id)  # create camera object
        cv2.namedWindow("PONIX", cv2.WND_PROP_FULLSCREEN)

        cam_width, cam_height = get_camera_shape(cam)
        print("cam_width: %d / cam_height: %d" % (cam_width, cam_height))

        width = width if width % 4 == 0 else width + 4 - (width % 4)  # must be divisible by 4
        height = int(
            width * float(cam_height / cam_width)
        )  # to keep the image ratio between width and height like camera
        height = height if height % 4 == 0 else height + 4 - (height % 4)  # must be divisible by 4
        print("width: %d / height: %d" % (width, height))

        # create style instance
        theme = my_utils.Theme(sess, height, width, models)

        # enter cam loop
        nm = 0
        while True:
            _, frame = cam.read()  # images from camera
            print("width: %d / height: %d" % (width, height))

            frame = cv2.resize(frame, (width, height))
            frame = cv2.flip(frame, 1)  # 1: horizontal reversal / 0: vertical reversal

            # X = np.zeros(self.batch_shape, dtype=np.float32)
            # X[0] = frame

            output = theme.get_output(frame)
            style = theme.get_style()

            # adjust ouput display
            if disp_source:
                full_img = make_triptych(disp_width, frame, style, output, horizontal)
                cv2.imshow("frame", full_img)
            else:
                print("not disp_source ")
                oh, ow, _ = output.shape
                output = cv2.resize(ouput, (disp_width, int(oh * disp_width / ow)))
                cv2.imshow("frame", output)

            # additional functions
            key_ = cv2.waitKeyEx(1)
            print(key_)

            if key_ == 27:  # [esc]: exit
                break
            elif key_ == 32:  # [spacebar]: stop video
                while True:
                    key2 = cv2.waitKeyEx(1)
                    cv2.imshow("frame", full_img)
                    if key2 == 32:  # [spacebar]: back to video
                        break
                    elif key2 == 65361:  # [left arrow]: previous style
                        theme.change_style(is_prev=True)
                        style = theme.get_style()
                        output = theme.get_output(frame)
                        full_img = make_triptych(disp_width, frame, style, output, horizontal)
                        cv2.imshow("frame", full_img)
                    elif key2 == 65363:  # [left arrow]: next style
                        theme.change_style(is_prev=False)
                        style = theme.get_style()
                        output = theme.get_output(frame)
                        full_img = make_triptych(disp_width, frame, style, output, horizontal)
                        cv2.imshow("frame", full_img)
                    # elif key2 == 13:  # [enter]: print out
                    #     img = cv2.resize(output, (5120, 3840))  # resizing
                    #     cv2.imwrite("./print" + "/" "print.png", img)
                    #     os.system("lpr ./print/print.png")
                    elif key2 == 13:  # [enter]: print out with phrase
                        pr_img = np.zeros((3840, 5120, 3), dtype="uint8") + 255  # resizing
                        resizeH = int(3840 * 1)
                        img = cv2.resize(output, (5120, resizeH))
                        pr_img[:resizeH, :, :] = img
                        img_pil = Image.fromarray(pr_img)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text(
                            (640, 3640),
                            "포항공과대학교 인공지능연구원 방문 기념  {}.{}.{}".format(year, month, day),
                            font=font,
                            fill=(255, 255, 255, 0),
                        )
                        pr_img = np.array(img_pil)
                        cv2.imwrite("./print" + "/" "print.png", pr_img)
                        os.system("lpr ./print/print.png")
            elif key_ == ord("c"):  # [c]: save capture image
                cv2.imwrite("./capture" + "/" + "./capture_%s.png" % nm, output)
                print("picture is saved!")
                nm += 1
            elif key_ == 65361:  # [left arrow]: previous style
                theme.change_style(is_prev=True)
                style = theme.get_style()
            elif key_ == 65363:  # [left arrow]: next style
                theme.change_style(is_prev=False)
                style = theme.get_style()
            # elif key_ == ord("p"):
            #     # 용지 크기에 맞게 resizing
            #     pr_img = cv2.resize(output, (5120, 3840))
            #     cv2.imwrite("./print" + "/" "print.png", pr_img)
            #     # os.system("lpr ./print/print.png")
            elif key_ == 13:  # [enter]: print out with phrase
                pr_img = np.zeros((3840, 5120, 3), dtype="uint8") + 255  # resizing
                resizeH = int(3840 * 1)
                img = cv2.resize(output, (5120, resizeH))
                pr_img[:resizeH, :, :] = img
                img_pil = Image.fromarray(pr_img)
                draw = ImageDraw.Draw(img_pil)
                draw.text(
                    (640, 3640),
                    "포항공과대학교 인공지능연구원 방문 기념  {}.{}.{}".format(year, month, day),
                    font=font,
                    fill=(255, 255, 255, 0),
                )
                pr_img = np.array(img_pil)
                cv2.imwrite("./print" + "/" "print.png", pr_img)
                os.system("lpr ./print/print.png")

            # change style automatically
            t2 = datetime.now()
            dt = t2 - t1
            if num_sec > 0 and dt.seconds > num_sec:
                t1 = datetime.now()
                theme.change_style(is_prev=False)
                style = theme.get_style()

        # done
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(
        opts["device_id"],
        opts["width"],
        opts["disp_width"],
        opts["disp_source"] == 1,
        opts["horizontal"] == 1,
        opts["num_sec"],
    )

# import packages
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import numpy as np
import cv2
import transform  # import run_test.py

import os, pdb, argparse
from datetime import date, datetime


date_obj = date.today()
year = date_obj.year
month = date_obj.month
day = date_obj.day


# config models
models = [
    {"path": "models/model_eunsook_batch_8/final.ckpt", "img": "style/eunsook.jpg", "artist": "Park Eun Sook",},
    {"path": "models/model_gohg/final.ckpt", "img": "style/gohg.jpg", "artist": "gohg",},
    {"path": "models/model_unknown02/final.ckpt", "img": "style/unknown02.jpg", "artist": "gohg",},
    {"path": "models/model_wood/final.ckpt", "img": "style/wood.jpg", "artist": "gohg",},
]


# add pr_width
opts = {
    "device_id": 0,
    "width": 700,
    "disp_width": 1250,
    "disp_source": 1,
    "horizontal": 1,
    "num_sec": 10,
    "pr_width": 10,
}

# load pre-trained model
def load_checkpoint(model_path, sess):
    saver = tf.train.Saver()
    try:
        saver.restore(sess, model_path)
        print("load checkpoint: ", model_path)
        # style = cv2.imread(model_path)
        return True
    except:
        print("checkpoint %s not loaded correctly" % model_path)
        return False


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


def main(device_id, width, disp_width, disp_source, horizontal, num_sec, pr_width):
    # define variables
    t1 = datetime.now()
    idx_model = 0  # index of model
    count = 0

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

        img_shape = (height, width, 3)
        batch_shape = (1,) + img_shape  # add one dim for batch

        # graph input
        x = tf.placeholder(tf.float32, shape=img_shape, name="input")
        x_batch = tf.expand_dims(x, 0)  # add one dim for batch

        # result image from transform
        trans = transform.Transform()
        preds = trans.net(x_batch / 255.0)
        preds = tf.squeeze(preds)  # remove one dim for batch
        preds = tf.clip_by_value(preds, 0.0, 255.0)

        # load checkpoint
        load_checkpoint(models[idx_model]["path"], sess)
        style = cv2.imread(models[idx_model]["img"])

        # enter cam loop
        nm = 0
        while True:
            _, frame = cam.read()  # images from camera
            print("width: %d / height: %d" % (width, height))

            frame = cv2.resize(frame, (width, height))
            frame = cv2.flip(frame, 1)  # 1: horizontal reversal / 0: vertical reversal

            X = np.zeros(batch_shape, dtype=np.float32)
            X[0] = frame

            output = sess.run(preds, feed_dict={x: X[0]})
            output = output[:, :, [2, 1, 0]].reshape(img_shape)
            output = np.clip(output, 0.0, 255.0).astype(np.uint8)
            output = cv2.resize(output, (width, height))

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
            key_ = cv2.waitKey(1)
            # exit
            if key_ == 27:
                break
            # 화면 멈춤 stop
            elif key_ == ord("s"):
                while True:
                    key2 = cv2.waitKey(1)
                    cv2.imshow("frame", full_img)

                    if key2 == ord("s"):
                        break

                    # stop before
                    elif key2 == ord("b"):
                        idx_model = (idx_model + len(models) - 1) % len(models)
                        # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                        load_checkpoint(models[idx_model]["path"], sess)
                        style = cv2.imread(models[idx_model]["img"])

                        output = sess.run(preds, feed_dict={x: X[0]})
                        output = output[:, :, [2, 1, 0]].reshape(img_shape)
                        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
                        output = cv2.resize(output, (width, height))

                        full_img = make_triptych(disp_width, frame, style, output, horizontal)
                        cv2.imshow("frame", full_img)

                    # stop after
                    elif key2 == ord("a"):
                        idx_model = (idx_model + 1) % len(models)
                        # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                        load_checkpoint(models[idx_model]["path"], sess)
                        style = cv2.imread(models[idx_model]["img"])

                        output = sess.run(preds, feed_dict={x: X[0]})
                        output = output[:, :, [2, 1, 0]].reshape(img_shape)
                        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
                        output = cv2.resize(output, (width, height))

                        full_img = make_triptych(disp_width, frame, style, output, horizontal)
                        cv2.imshow("frame", full_img)
                    # stop print

                    elif key2 == ord("p"):
                        # resizing
                        img = cv2.resize(output, (5120, 3840))
                        cv2.imwrite("./print" + "/" "print.png", img)
                        os.system("lpr ./print/print.png")

                    elif key2 == ord("o"):
                        # resizing
                        pr_img = np.zeros((3840, 5120, 3), dtype="uint8") + 255
                        resizeH = int(3840 * 1)
                        img = cv2.resize(output, (5120, resizeH))
                        pr_img[:resizeH, :, :] = img
                        img_pil = Image.fromarray(pr_img)
                        draw = ImageDraw.Draw(img_pil)
                        # draw.text((640,3640), '포항공과대학교 인공지능대학원·연구원 방문 기념  2020.09.11', font=font, fill=(255,255,255,0))
                        draw.text(
                            (640, 3640),
                            "포항공과대학교 인공지능연구원 방문 기념  {}.{}.{}".format(year, month, day),
                            font=font,
                            fill=(255, 255, 255, 0),
                        )
                        pr_img = np.array(img_pil)
                        cv2.imwrite("./print" + "/" "print.png", pr_img)
                        # cv2.imwrite('./print' + '/' 'print_frame{}.png'.format(count), frame)
                        # count+=1
                        os.system("lpr ./print/print.png")

            # 다음 테마로 가기 after
            elif key_ == ord("a"):
                idx_model = (idx_model + 1) % len(models)
                # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["path"], sess)
                style = cv2.imread(models[idx_model]["img"])

            # 해당 화면의 결과 캡처 및 저장 capture
            elif key_ == ord("c"):
                cv2.imwrite("./capture" + "/" + "./capture_%s.png" % nm, output)
                print("picture is saved!")
                nm += 1

            # 이전 테마로 되돌아가기 before
            elif key_ == ord("b"):
                idx_model = (idx_model + len(models) - 1) % len(models)
                # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["path"], sess)
                style = cv2.imread(models[idx_model]["img"])

            # 다음 테마로 가기 after
            elif key_ == ord("a"):
                idx_model = (idx_model + 1) % len(models)
                # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["path"], sess)
                style = cv2.imread(models[idx_model]["img"])
            # 프린트 prints

            elif key_ == ord("p"):
                # 용지 크기에 맞게 resizing
                pr_img = cv2.resize(output, (5120, 3840))
                cv2.imwrite("./print" + "/" "print.png", pr_img)
                # os.system("lpr ./print/print.png")

            elif key_ == ord("o"):
                # resizing
                pr_img = np.zeros((3840, 5120, 3), dtype="uint8") + 255
                resizeH = int(3840 * 1)
                img = cv2.resize(output, (5120, resizeH))
                pr_img[:resizeH, :, :] = img
                img_pil = Image.fromarray(pr_img)
                draw = ImageDraw.Draw(img_pil)
                # draw.text((640,3640), '포항공과대학교 인공지능대학원·연구원 방문 기념  2020.09.11', font=font, fill=(255,255,255,0))
                draw.text(
                    (640, 3640),
                    "포항공과대학교 인공지능연구원 방문 기념  {}.{}.{}".format(year, month, day),
                    font=font,
                    fill=(255, 255, 255, 0),
                )
                pr_img = np.array(img_pil)
                cv2.imwrite("./print" + "/" "print.png", pr_img)
                # cv2.imwrite('./print' + '/' 'print_frame{}.png'.format(count), frame)
                # count+=1
                os.system("lpr ./print/print.png")

            t2 = datetime.now()
            dt = t2 - t1

            if num_sec > 0 and dt.seconds > num_sec:
                t1 = datetime.now()
                idx_model = (idx_model + 1) % len(models)
                # print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["path"], sess)
                style = cv2.imread(models[idx_model]["img"])

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
        opts["pr_width"],
    )

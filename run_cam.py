from PIL import Image, ImageFont, ImageDraw, ImageTk
from datetime import date, datetime

import tkinter as tk
import tensorflow as tf
import numpy as np

import cv2, imutils
import csv
import os, sys
import transform, cam_utils


class App(tk.Frame):
    def __init__(self, master, opts):
        self.master = master

        """ config master window """
        self.master.title("PONIX")  # set title
        self.master.attributes("-zoomed", True)  # initialize a window as maximized
        self.master.resizable(True, True)  # allow to resize
        self.font_ms_serif = tk.font.Font(self.master, family="MS Serif", size=12)  # config font

        """ config frame """
        self.frame_top = tk.Frame(self.master)
        self.frame_top.pack(side=tk.TOP, fill=tk.X)
        self.frame_left = tk.Frame(self.master)
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y)
        self.frame_right = tk.Frame(self.master)
        self.frame_right.pack(side=tk.RIGHT)

        """ config button """
        self.btn_prev = tk.Button(self.frame_top, width=10, text="◀", command=lambda: change_style(True))
        self.btn_prev.pack(side=tk.LEFT)
        self.btn_stop = tk.Button(self.frame_top, width=10, text="||", command=self.video_stop)
        self.btn_stop.pack(side=tk.LEFT)
        self.btn_next = tk.Button(self.frame_top, width=10, text="▶", command=lambda: change_style(False))
        self.btn_next.pack(side=tk.LEFT)
        self.btn_capture = tk.Button(self.frame_top, width=20, text="Capture", command=self.capture)
        self.btn_capture.pack(side=tk.LEFT)
        self.btn_print = tk.Button(self.frame_top, width=20, text="Print", command=self.print_out, state=tk.DISABLED)
        self.btn_print.pack(side=tk.LEFT)
        self.btn_save = tk.Button(self.frame_top, width=20, text="Save", command=self.save, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT)

        """ config text """
        self.text_input = tk.Text(self.frame_top, height=2, font=self.font_ms_serif)
        self.text_input.pack(side=tk.LEFT)
        self.text_artist = tk.Label(
            self.frame_top,
            font=self.font_ms_serif,
            text=("「" + models[theme.idx]["title"] + "」" + ", " + models[theme.idx]["artist"]),
        )
        self.text_artist.pack(side=tk.LEFT, padx=10)

        """ config label """
        self.label_content = tk.Label(self.frame_left)
        self.label_content.grid()
        self.label_output = tk.Label(self.frame_right)
        self.label_output.grid(row=0, column=0)
        self.label_style = tk.Label(self.frame_left)
        self.label_style.grid(row=1, column=0)

        """ config member variable """
        self.IS_VIDEO_STOP = False
        with open("final_model_list.csv", "r") as f:
            self.models = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]

        """ play """
        self.open_tf_session()
        self.video_play()

    def open_tf_session(self):
        g = tf.Graph()
        soft_config = tf.ConfigProto(allow_soft_placement=True)
        soft_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=soft_config)

        self.cam = cv2.VideoCapture(opts.get("device_id", 0))
        cam_width, cam_height = get_camera_shape(self.cam)
        width = width if width % 4 == 0 else width + 4 - (width % 4)  # must be divisible by 4
        height = int(
            width * float(cam_height / cam_width)
        )  # to keep the image ratio between width and height like camera
        height = height if height % 4 == 0 else height + 4 - (height % 4)  # must be divisible by 4

    def get_camera_shape(self, cam):
        cv_version_major, _, _ = cv2.__version__.split(".")
        print("cv_version_major: ", cv_version_major)
        if cv_version_major == "3" or cv_version_major == "4":
            return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    def get_date(self):
        date_obj = date.today()
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        # TODO: return date string

    def video_play(self):
        if IS_VIDEO_STOP == False:
            ret, frame = cam.read()

            if not ret:
                print("could not receive frame")
                cam.release()
                return

            self.frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.flip(frame, 1)
            output = theme.get_output(frame)

            ch, cw, _ = frame.shape  # cam shape
            oh, ow, _ = output.shape  # output display shape

            disp_height = int(disp_width * (oh / ow))
            # h = int((ch / cw) * disp_width * 0.5)
            # w = int((cw / ch) * disp_height * 0.5)
            w = int(root.winfo_screenwidth() / 2 - cam_width)
            h = int(w * (oh / ow))
            # print("w, h", disp_width, disp_height)

            content = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            output = Image.fromarray(cv2.cvtColor(cv2.resize(output, (disp_width, disp_height)), cv2.COLOR_BGR2RGB))
            # style = Image.open(models[theme.idx]["img"]).resize((700, 528), Image.ANTIALIAS)
            style = theme.get_style()
            # print(style.shape[0], style.shape[1])  # shape 0 : height
            style = (
                imutils.resize(style, height=500)
                if (style.shape[1] / style.shape[0]) * 500 < 700
                else imutils.resize(style, width=700)
            )
            style = Image.fromarray(cv2.cvtColor(style, cv2.COLOR_BGR2RGB))

            imgtk_content = ImageTk.PhotoImage(image=content)
            imgtk_output = ImageTk.PhotoImage(image=output)
            imgtk_style = ImageTk.PhotoImage(image=style)

            # opencv video
            label_content.imgtk = imgtk_content
            label_content.configure(image=imgtk_content)
            label_output.imgtk = imgtk_output
            label_output.configure(image=imgtk_output)
            label_style.imgtk = imgtk_style
            label_style.configure(image=imgtk_style)

            root.after(1, video_play)
        else:
            output = theme.get_output(frame)

            ch, cw, _ = frame.shape  # cam shape
            oh, ow, _ = output.shape  # output display shape

            disp_height = int(disp_width * (oh / ow))
            h = int((ch / cw) * disp_width * 0.5)
            w = int((cw / ch) * disp_height * 0.5)

            content = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            output = Image.fromarray(cv2.cvtColor(cv2.resize(output, (disp_width, disp_height)), cv2.COLOR_BGR2RGB))
            style = theme.get_style()
            # print(style.shape[0], style.shape[1])  # shape 0 : height
            style = (
                imutils.resize(style, height=500)
                if (style.shape[1] / style.shape[0]) * 500 < 700
                else imutils.resize(style, width=700)
            )
            style = Image.fromarray(cv2.cvtColor(style, cv2.COLOR_BGR2RGB))

            imgtk_content = ImageTk.PhotoImage(image=content)
            imgtk_output = ImageTk.PhotoImage(image=output)
            imgtk_style = ImageTk.PhotoImage(image=style)

            # opencv video
            label_content.imgtk = imgtk_content
            label_content.configure(image=imgtk_content)
            label_output.imgtk = imgtk_output
            label_output.configure(image=imgtk_output)
            label_style.imgtk = imgtk_style
            label_style.configure(image=imgtk_style)
            return

    def video_stop(self):
        global IS_VIDEO_STOP
        IS_VIDEO_STOP = not IS_VIDEO_STOP

        if IS_VIDEO_STOP == False:
            btn_print["state"] = tk.DISABLED
            btn_save["state"] = tk.DISABLED
        else:
            # btn_print["state"] = tk.NORMAL
            btn_save["state"] = tk.NORMAL

        # remove all print files
        if os.path.exists("./print/print.png"):
            os.remove(r"./print/print.png")
        video_play()

    def capture(self):
        global nm, frame
        cv2.imwrite("./capture" + "/" + "./capture_%s.png" % nm, theme.get_output(frame))
        # print("picture is saved!")
        nm += 1

    def save(self):
        global frame
        output = theme.get_output(frame)

        ch, cw, _ = frame.shape  # cam shape
        oh, ow, _ = output.shape  # output display shape

        disp_height = int(disp_width * (oh / ow))
        h = int((ch / cw) * disp_width * 0.5)
        w = int((cw / ch) * disp_height * 0.5)

        """
        here to start the merge output & style image
        """
        style_downscale = imutils.resize(theme.get_style(), width=100, inter=cv2.INTER_AREA)
        x_offset = 590
        y_offset = 10
        output[
            y_offset : y_offset + style_downscale.shape[0], x_offset : x_offset + style_downscale.shape[1]
        ] = style_downscale

        # write text
        text = text_input.get("1.0", tk.END)

        pr_img = np.zeros((3840, 5120, 3), dtype="uint8") + 255  # resizing
        resizeH = int(3840 * 1)
        img = cv2.resize(output, (5120, resizeH))
        pr_img[:resizeH, :, :] = img
        img_pil = Image.fromarray(pr_img)

        draw = ImageDraw.Draw(img_pil)
        draw.text((100, 3550), text, font=font, fill=(255, 255, 255))
        output = np.array(img_pil)

        # TODO: here to save
        # save to file for printing
        # cv2.imwrite("./print/print.png", output)

        """
        update the label image
        """
        cv2.imwrite("./print/print.png", cv2.resize(output, (disp_width, disp_height)))
        output = Image.fromarray(cv2.cvtColor(cv2.resize(output, (disp_width, disp_height)), cv2.COLOR_BGR2RGB))
        imgtk_output = ImageTk.PhotoImage(image=output)

        # opencv video
        label_output.imgtk = imgtk_output
        label_output.configure(image=imgtk_output)

        btn_print["state"] = tk.NORMAL

    def print_out(self):
        # global frame
        # output = theme.get_output(frame)
        # pr_img = np.zeros((3840, 5120, 3), dtype="uint8") + 255  # resizing
        # resizeH = int(3840 * 1)
        # img = cv2.resize(output, (5120, resizeH))
        # pr_img[:resizeH, :, :] = img
        # img_pil = Image.fromarray(pr_img)
        # draw = ImageDraw.Draw(img_pil)
        # draw.text(
        #     (640, 3640), "포항공과대학교 인공지능연구원 방문 기념  {}.{}.{}".format(year, month, day), font=font, fill=(255, 255, 255, 0),
        # )
        # pr_img = np.array(img_pil)
        # cv2.imwrite("./print" + "/" "print.png", pr_img)
        os.system("lpr ./print/print.png")

    def change_style(self, is_prev=True):
        global IS_VIDEO_STOP
        theme.change_style(is_prev)
        text_artist.configure(text="「" + models[theme.idx]["title"] + "」" + ", " + models[theme.idx]["artist"])
        if IS_VIDEO_STOP == True:
            video_play()


if __name__ == "__main__":
    opts = {
        "device_id": 0,
        "width": 700,
        "disp_width": 1215,
        "disp_source": 1,
        "horizontal": 1,
        "num_sec": 10,
    }
    root = tk.Tk()
    app = App(root, opts)
    root.mainloop()


# config font
# fontpath = "/usr/share/fonts/truetype/nanum/NanumBrush.ttf"  # brush
fontpath = "/usr/share/fonts/truetype/nanum/NanumPen.ttf"  # pen
font = ImageFont.truetype(font=fontpath, size=250)
###################################################################

# create style instance
theme = cam_utils.Theme(sess, height, width, models)

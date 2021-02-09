from PIL import Image, ImageFont, ImageDraw, ImageTk
import tkinter as tk
import tkinter.font as tkFont
import tensorflow as tf
import numpy as np
import cv2, imutils
import transform, my_utils  # import transform.py, cam_utils.py


import os, sys
from datetime import date, datetime

### define variables
date_obj = date.today()
year = date_obj.year
month = date_obj.month
day = date_obj.day
# if os.environ.get("DISPLAY", "") == "":
#     print("no display found. Using :0.0")
#     os.environ.__setitem__("DISPLAY", ":0.0")


def get_camera_shape(cam):
    cv_version_major, _, _ = cv2.__version__.split(".")
    print("cv_version_major: ", cv_version_major)
    if cv_version_major == "3" or cv_version_major == "4":
        return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


models = [
    {"path": "models/model_gohg/final.ckpt", "img": "style/gohg.jpg", "artist": "gohg",},
    {"path": "models/model_unknown02/final.ckpt", "img": "style/unknown02.jpg", "artist": "unknown",},
    {"path": "models/model_wood/final.ckpt", "img": "style/wood.jpg", "artist": "wood",},
]

opts = {
    "device_id": 0,
    "width": 700,
    "disp_width": 1250,
    "disp_source": 1,
    "horizontal": 1,
    "num_sec": 10,
}


# config font
# fontpath = "/usr/share/fonts/truetype/nanum/NanumBrush.ttf"  # brush
fontpath = "/usr/share/fonts/truetype/nanum/NanumPen.ttf"  # pen
font = ImageFont.truetype(font=fontpath, size=200)
###################################################################

# create and config tkinter instance
# print(list(tkFont.families()))

root = tk.Tk()
root.title("PONIX")
root.attributes("-zoomed", True)
root.resizable(True, True)
tk_font = tkFont.Font(root, family="MS Serif", size=12)
# root.configure(bg="black")

width = 700  # content size
device_id = 0
disp_width = 1215
disp_source = 1
horizontal = 1
num_sec = 1


# open tensorflow session
g = tf.Graph()
soft_config = tf.ConfigProto(allow_soft_placement=True)
soft_config.gpu_options.allow_growth = True
sess = tf.Session(config=soft_config)

cam = cv2.VideoCapture(device_id)
cam_width, cam_height = get_camera_shape(cam)
print(cam_width, cam_height)
width = width if width % 4 == 0 else width + 4 - (width % 4)  # must be divisible by 4
height = int(width * float(cam_height / cam_width))  # to keep the image ratio between width and height like camera
height = height if height % 4 == 0 else height + 4 - (height % 4)  # must be divisible by 4


# create style instance
theme = my_utils.Theme(sess, height, width, models)


# scrollbar = tk.Scrollbar(root, orient="vertical")
# scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
IS_VIDEO_STOP = False
frame = None
nm = 0

# TODO if stop the video, then init the print directory -> and then activate print, save button!
def video_stop():
    print("hello")
    global IS_VIDEO_STOP
    IS_VIDEO_STOP = not IS_VIDEO_STOP

    if IS_VIDEO_STOP == False:
        btn_print["state"] = tk.DISABLED
        btn_save["state"] = tk.DISABLED
    else:
        btn_print["state"] = tk.NORMAL
        btn_save["state"] = tk.NORMAL

    # remove all print files
    if os.path.exists("./print/print.png"):
        os.remove(r"./print/print.png")
    video_play()


def capture():
    global nm, frame
    cv2.imwrite("./capture" + "/" + "./capture_%s.png" % nm, theme.get_output(frame))
    print("picture is saved!")
    nm += 1


def save():
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
    draw.text((100, 3630), text, font=font, fill=(255, 255, 255))
    output = np.array(img_pil)

    # save to file for printing
    cv2.imwrite("./print/print.png", output)

    """
    update the label image
    """
    output = Image.fromarray(cv2.cvtColor(cv2.resize(output, (disp_width, disp_height)), cv2.COLOR_BGR2RGB))
    imgtk_output = ImageTk.PhotoImage(image=output)

    # opencv video
    label_output.imgtk = imgtk_output
    label_output.configure(image=imgtk_output)


def print_out():
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


# config frames
frame_top = tk.Frame(root)
frame_top.pack(side=tk.TOP, fill=tk.X)
frame_left = tk.Frame(root)
frame_left.pack(side=tk.LEFT, fill=tk.Y)
frame_right = tk.Frame(root)
frame_right.pack(side=tk.RIGHT)

# config widgets
btn_prev = tk.Button(frame_top, width=10, text="◀", command=lambda: change_style(True))
btn_prev.pack(side=tk.LEFT)
btn_stop = tk.Button(frame_top, width=10, text="||", command=video_stop)
btn_stop.pack(side=tk.LEFT)
btn_next = tk.Button(frame_top, width=10, text="▶", command=lambda: change_style(False))
btn_next.pack(side=tk.LEFT)
btn_capture = tk.Button(frame_top, width=20, text="Capture", command=capture)
btn_capture.pack(side=tk.LEFT)

# print button
btn_print = tk.Button(frame_top, width=20, text="Print", command=print_out, state=tk.DISABLED)
btn_print.pack(side=tk.LEFT)
btn_save = tk.Button(frame_top, width=20, text="Save", command=save, state=tk.DISABLED)
btn_save.pack(side=tk.LEFT)
text_input = tk.Text(frame_top, height=2, font=tk_font)
text_input.pack(side=tk.LEFT)
text_artist = tk.Label(frame_top, font=tk_font, text=models[theme.idx]["artist"])
text_artist.pack(side=tk.LEFT, padx=10)

label_content = tk.Label(frame_left)
label_content.grid()
label_output = tk.Label(frame_right)
label_output.grid(row=0, column=0)
label_style = tk.Label(frame_left)
label_style.grid(row=1, column=0)


def video_play():
    global frame
    if IS_VIDEO_STOP == False:
        ret, frame = cam.read()
        if not ret:
            print("could not receive frame")
            cam.release()
            return

        frame = cv2.resize(frame, (width, height))
        frame = cv2.flip(frame, 1)
        output = theme.get_output(frame)

        ch, cw, _ = frame.shape  # cam shape
        oh, ow, _ = output.shape  # output display shape

        disp_height = int(disp_width * (oh / ow))
        # h = int((ch / cw) * disp_width * 0.5)
        # w = int((cw / ch) * disp_height * 0.5)
        w = int(root.winfo_screenwidth() / 2 - cam_width)
        h = int(w * (oh / ow))
        print("w, h", disp_width, disp_height)

        content = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = Image.fromarray(cv2.cvtColor(cv2.resize(output, (disp_width, disp_height)), cv2.COLOR_BGR2RGB))
        # style = Image.open(models[theme.idx]["img"]).resize((700, 528), Image.ANTIALIAS)
        style = Image.fromarray(cv2.cvtColor(imutils.resize(theme.get_style(), height=500), cv2.COLOR_BGR2RGB))

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
        style = Image.fromarray(cv2.cvtColor(imutils.resize(theme.get_style(), height=500), cv2.COLOR_BGR2RGB))

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


def change_style(is_prev=True):
    global IS_VIDEO_STOP
    theme.change_style(is_prev)
    text_artist.configure(text=models[theme.idx]["artist"])
    if IS_VIDEO_STOP == True:
        video_play()


video_play()
root.mainloop()  # start gui

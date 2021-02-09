from PIL import Image, ImageFont, ImageDraw, ImageTk
import tkinter as tk
import tkinter.font as tkFont
import tensorflow as tf
import numpy as np
import cv2
import transform, my_utils  # import transform.py, cam_utils.py


import os, sys

if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using :0.0")
    os.environ.__setitem__("DISPLAY", ":0.0")


def get_camera_shape(cam):
    cv_version_major, _, _ = cv2.__version__.split(".")
    print("cv_version_major: ", cv_version_major)
    if cv_version_major == "3" or cv_version_major == "4":
        return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


models = [
    {"path": "models/model_eunsook_batch_8/final.ckpt", "img": "style/eunsook.jpg", "artist": "Park Eun Sook",},
    {"path": "models/model_gohg/final.ckpt", "img": "style/gohg.jpg", "artist": "gohg",},
    {"path": "models/model_unknown02/final.ckpt", "img": "style/unknown02.jpg", "artist": "gohg",},
    {"path": "models/model_wood/final.ckpt", "img": "style/wood.jpg", "artist": "gohg",},
]

opts = {
    "device_id": 0,
    "width": 700,
    "disp_width": 1250,
    "disp_source": 1,
    "horizontal": 1,
    "num_sec": 10,
}
width = 700
device_id = 0
disp_width = 1250
disp_source = 1
horizontal = 1
num_sec = 1

# config font
fontpath = "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf"
font = ImageFont.truetype(font=fontpath, size=150)
###################################################################

# create and config tkinter instance
# print(list(tkFont.families()))

root = tk.Tk()
root.title("PONIX")
root.attributes("-zoomed", True)
root.resizable(True, True)
tk_font = tkFont.Font(root, family="MS Serif", size=12)


def get_text_input():
    # TODO: get text on the rootdow
    pass


def print_with_text():
    text = text_input.get("1.0", "end")
    # TODO: imwrite() with text


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


# scrollbar = tk.Scrollbar(root, orient="vertical")
# scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


# config frames
frame_top = tk.Frame(root, bg="cyan")
frame_top.pack(side=tk.TOP, fill=tk.X)
frame_left = tk.Frame(root, bg="white")
frame_left.pack(side=tk.LEFT, fill=tk.Y)
frame_right = tk.Frame(root, bg="black")
frame_right.pack(side=tk.RIGHT)

# config widgets
btn_prev = tk.Button(frame_top, width=10, text="◀")
btn_prev.pack(side=tk.LEFT)
btn_stop = tk.Button(frame_top, width=20, text="||")
btn_stop.pack(side=tk.LEFT)
btn_next = tk.Button(frame_top, width=10, text="▶")
btn_next.pack(side=tk.LEFT)
btn_print = tk.Button(frame_top, width=20, text="Print")
btn_print.pack(side=tk.LEFT)
# btn_preview = tk.Button(frame_top, width=20, text="Preview", command=get_text_input)
text_input = tk.Text(frame_top, height=2, font=tk_font)
text_input.pack(side=tk.LEFT)

# TODO: widget position setting
# btn_prev.grid(row=1, column=0)
# btn_next.grid(row=1, column=1)
# btn_print.grid()
# btn_text.grid()

label_content = tk.Label(frame_right)
label_content.grid()
label_output = tk.Label(frame_left)
label_output.grid(row=0, column=0)
label_style = tk.Label(frame_left)
label_style.grid(row=1, column=0)

# create style instance
theme = my_utils.Theme(sess, height, width, models)


def video_play():
    ret, frame = cam.read()
    if not ret:
        print("could not receive frame")
        cam.release()
        return

    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)

    content = Image.fromarray(frame)
    output = Image.fromarray(theme.get_output(frame))
    style = Image.open("./style/eunsook.jpg").resize((700, 528), Image.ANTIALIAS)

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


video_play()
root.mainloop()  # start gui

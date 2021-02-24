from PIL import Image, ImageFont, ImageDraw, ImageTk
from datetime import date, datetime

from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.utils import COMMASPACE, formatdate

import tkinter as tk
import tkinter.font as tkFont
import tkinter.messagebox as tkMassage
import smtplib
from tkinter import simpledialog
from tkinter import messagebox

import tensorflow as tf
import numpy as np

import cv2, imutils
import csv
import os, sys, argparse
import transform, cam_utils
import ctypes
import time

class App(tk.Frame):
    def __init__(self, master, args):
        """ config master window """
        self.master = master
        self.master.attributes("-zoomed", True)  # initialize window as maximized
        self.master.title("PONIX")  # set title
        self.master.resizable(False, False)  # not allow to resize
        # TODO: change font
        self.font_ms_serif = tkFont.Font(self.master, family="MS Serif", size=12)  # config font
        print("available font list: ", list(tkFont.families()))

        """ config frame """
        self.frame_top = tk.Frame(self.master)
        self.frame_top.pack(side=tk.TOP, fill=tk.X)
        self.frame_left = tk.Frame(self.master)
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y)
        self.frame_right = tk.Frame(self.master)
        self.frame_right.pack(side=tk.RIGHT)

        """ config button """
        self.btn_prev = tk.Button(self.frame_top, width=5, text="◀", command=lambda: self.change_style(True))
        self.btn_prev.pack(side=tk.LEFT)
        self.btn_stop = tk.Button(self.frame_top, width=5, text="||", command=self.video_stop)
        self.btn_stop.pack(side=tk.LEFT)
        self.btn_next = tk.Button(self.frame_top, width=5, text="▶", command=lambda: self.change_style(False))
        self.btn_next.pack(side=tk.LEFT)
        self.btn_capture = tk.Button(self.frame_top, width=15, text="Capture", command=self.capture)
        self.btn_capture.pack(side=tk.LEFT)
        self.btn_print = tk.Button(self.frame_top, width=15, text="Print", command=self.print_out, state=tk.DISABLED)
        self.btn_print.pack(side=tk.LEFT)
        self.btn_save = tk.Button(self.frame_top, width=15, text="Save", command=self.save, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT)

        """ config text """
        self.text_input = tk.Text(self.frame_top, height=1, font=self.font_ms_serif)
        self.text_input.pack(side=tk.LEFT)
        self.text_artist = tk.Label(self.frame_top, font=self.font_ms_serif, text="")
        self.text_artist.pack(side=tk.RIGHT, padx=10)

        """ config label """
        self.label_content = tk.Label(self.frame_left)
        self.label_content.grid()
        self.label_output = tk.Label(self.frame_right)
        self.label_output.grid(row=0, column=0)
        self.label_style = tk.Label(self.frame_left)
        self.label_style.grid(row=1, column=0)

        """ config member variable """
        self.master.update()
        with open(args.models, "r") as f:
            self.models = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
        self.is_video_stop = False  # flag in order to check if video stop
        self.disp_height = (
            self.master.winfo_height() - self.frame_top.winfo_height()
        )  # height of display (not include navigation bar)
        self.cam = cam_utils.Cam(
            args.device_id, args.inp_width, self.master.winfo_width(), self.disp_height
        )  # cam instance
        self.styleTransfer = cam_utils.StyleTransfer(
            self.cam.inp_height, self.cam.inp_width, self.models
        )  # style transfer instance
        self.start_time = time.time()
            #time instance for autoplay
        self.sec = args.num_sec
            #time unit for style change in autoplay

        """ config email service """
        if args.email == True:
            load_dotenv()  # load env variable

            # set email address and password
            self.EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
            self.EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

            # config button for eamil service
            self.btn_email = tk.Button(
                self.frame_top, width=15, text="Email", command=self.send_email, state=tk.DISABLED
            )
            self.btn_email.pack(side=tk.LEFT)

        """ play """
        self.video_play()

    # update single label
    def update_label(self, bgr_img, label):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        content = Image.fromarray(rgb_img)
        imgtk_content = ImageTk.PhotoImage(image=content)
        label.imgtk = imgtk_content
        label.configure(image=imgtk_content)

    # update entire window
    def update_window(self, output=None):
        
        inp_frame, lab_frame = self.cam.get_frame()
        output = self.styleTransfer.get_output(inp_frame)
        style = self.styleTransfer.get_style()

        # resize output and style
        output = imutils.resize(output, height=self.disp_height)
        style = (
            cv2.resize(style, (lab_frame.shape[1], self.disp_height - lab_frame.shape[0]))
            # TODO: resize the style image in order to make them full size
            # if (style.shape[1] / style.shape[0]) * 500 < 700
            # else imutils.resize(style, width=700)
        )

        # load images in label
        self.update_label(lab_frame, self.label_content)
        self.update_label(output, self.label_output)
        self.update_label(style, self.label_style)

        # load style info in text
        self.text_artist.configure(text=self.styleTransfer.get_style_info())
   
    def video_play(self):
        start_time = time.time()
        if self.is_video_stop == True:
            self.update_window()
            return
        else:
            change_point = (start_time - self.start_time)%self.sec
            self.cam.set_frame()
            self.update_window()
            self.master.after(1,self.video_play)
            if change_point < 0.1:
                self.styleTransfer.change_style()
                #automatic style change

    def video_stop(self):
        self.is_video_stop = not self.is_video_stop

        if self.is_video_stop == True:
            self.btn_save["state"] = tk.NORMAL
        else:
            self.btn_print["state"] = tk.DISABLED
            self.btn_save["state"] = tk.DISABLED

            # for email service
            try:
                self.btn_email["state"] = tk.DISABLED
            except:
                pass
            self.video_play()

        # remove all print files
        if os.path.exists("./print/print.png"):
            os.remove(r"./print/print.png")

    def change_style(self, is_prev=True):
        self.styleTransfer.change_style(is_prev)
        if self.is_video_stop == True:
            self.video_play()
            
            
    def capture(self):
        inp_frame, _ = self.cam.get_frame()
        # e.g. capture_20210129_12'34'56
        cv2.imwrite(
            "capture/capture_%s.png" % datetime.now().strftime("%Y%m%d_%H'%M'%S"),
            self.styleTransfer.get_output(inp_frame),
        )
        print("save capture!")

    def save(self):
        inp_frame, _ = self.cam.get_frame()
        output = self.styleTransfer.get_output(inp_frame)

        oh, ow, _ = output.shape
        disp_height = self.disp_height
        disp_width = int(self.disp_height * (ow / oh))

        # merge ouput and style
        style_downscale = imutils.resize(self.styleTransfer.get_style(), width=100, inter=cv2.INTER_AREA)
        x_offset = 590
        y_offset = 10
        output[
            y_offset : y_offset + style_downscale.shape[0], x_offset : x_offset + style_downscale.shape[1]
        ] = style_downscale

        # write text
        text = self.text_input.get("1.0", tk.END)
        pr_img = np.zeros((3840, 5120, 3), dtype="uint8") + 255  # resizing
        resized_height = int(3840 * 1)
        img = cv2.resize(output, (5120, resized_height))
        pr_img[:resized_height, :, :] = img
        img_pil = Image.fromarray(pr_img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(
            (100, 3550),
            text,
            font=ImageFont.truetype(font="/usr/share/fonts/truetype/nanumfont/NanumPen.ttf", size=250),
            fill=(255, 255, 255),
        )
        
        # write date with postech sign
        today = datetime.now().strftime("%Y.%m.%d")
        text = today + ' POSTECH인공지능연구원'
        font = ImageFont.truetype('NanumGothicBold.ttf',90)
        draw.text((100,3430), text, font=font)
        

        # resizing
        output = np.array(img_pil)
        output = cv2.resize(output, (disp_width, disp_height))

        cv2.imwrite("print/print.png", output)  # save print.png
        self.update_label(output, self.label_output)  # update label
        self.btn_print["state"] = tk.NORMAL  # update button state

        # for email service
        try:
            self.btn_email["state"] = tk.NORMAL
        except:
            pass

    def print_out(self):
        os.system("lpr print/print.png")

    # send mail to respected receiver
    def send_email(self):
        sender = self.EMAIL_ADDRESS
        password = self.EMAIL_PASSWORD
        receiver = simpledialog.askstring(title="send email", prompt="your email address:")
        title = "from postech ai lab"
        filename = "photo.png"

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = COMMASPACE.join(receiver)
        msg["DATE"] = formatdate(localtime=True)
        msg["Subject"] = title
        # msg.attach(MIMEText())

        with open("print/print.png", "rb") as f:
            part = MIMEApplication(f.read(), Name=filename)

        part["Content-Disposition"] = 'attachment; filename="%s"' % filename
        msg.attach(part)

        try:
            # only for gmail account
            with smtplib.SMTP("smtp.gmail.com:587") as server:
                server.ehlo()  # local host
                server.starttls()  # put connection to smtp server
                server.login(sender, password)  # login to account of sender
                server.sendmail(sender, receiver, msg.as_string())
                server.close()
                print("success to send email", receiver)
                messagebox.showinfo(message="success to send email!")
        except Exception as e:
            print("fail to send mail:", e)
            messagebox.showerror(message="fail to send mail...(unexpected error occured)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", default=0, type=int, help="id of camera device")
    parser.add_argument("--inp_width", default=700, type=int, help="width of input image")
    # parser.add_argument("--disp_width", default=1215, type=int, help="width of window area")
    parser.add_argument("--num_sec", default=10, type=int, help="autoplay interval")
    parser.add_argument("--email", default=False, type=bool, help="want email service")
    parser.add_argument("--models", type=str, required=True, help="path/to/models.csv")
    return parser.parse_args()


def main():
    args = parse_args()  # parse arguments

    # create directory if not exist
    directories = ["capture", "print"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # show up tkinter window
    root = tk.Tk()
    app = App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

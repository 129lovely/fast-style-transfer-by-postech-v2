from PIL import Image, ImageFont, ImageDraw, ImageTk
from datetime import date, datetime

import tkinter as tk
import tensorflow as tf
import numpy as np

import cv2, imutils
import csv
import os, sys, argparse
import transform, cam_utils


class App(tk.Frame):
    def __init__(self, master, args):
        """ config member variable """
        self.is_video_stop = False
        with open(args.get("models"), "r") as f:
            self.models = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
        self.cam = cam_utils.Cam(args.device_id, args.width)
        self.st = cam_utils.StyleTransfer(self.cam.height, self.cam.width, self.models)
        self.disp_width = args.disp_width

        """ config master window """
        self.master = master
        self.master.title("PONIX")  # set title
        self.master.attributes("-zoomed", True)  # initialize window as maximized
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
        self.btn_prev = tk.Button(self.frame_top, width=10, text="◀", command=lambda: self.st.change_style(True))
        self.btn_prev.pack(side=tk.LEFT)
        self.btn_stop = tk.Button(self.frame_top, width=10, text="||", command=self.video_stop)
        self.btn_stop.pack(side=tk.LEFT)
        self.btn_next = tk.Button(self.frame_top, width=10, text="▶", command=lambda: self.st.change_style(False))
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
            text=("「" + self.models[st.idx]["title"] + "」" + ", " + self.models[st.idx]["artist"]),
        )
        self.text_artist.pack(side=tk.LEFT, padx=10)

        """ config label """
        self.label_content = tk.Label(self.frame_left)
        self.label_content.grid()
        self.label_output = tk.Label(self.frame_right)
        self.label_output.grid(row=0, column=0)
        self.label_style = tk.Label(self.frame_left)
        self.label_style.grid(row=1, column=0)

        """ play """
        self.video_play()

    def config_label(self):
        frame = self.cam.get_frame()
        output = self.st.get_output(frame)
        style = self.st.get_style()

        # get disp_height
        oh, ow, _ = output.shape
        disp_height = int(self.disp_width * (oh / ow))

        # resize style
        style = (
            imutils.resize(style, height=500)
            if (style.shape[1] / style.shape[0]) * 500 < 700
            else imutils.resize(style, width=700)
        )

        # load content
        content = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk_content = ImageTk.PhotoImage(image=content)
        self.label_content.imgtk = imgtk_content
        self.label_content.configure(image=imgtk_content)

        # load output
        output = Image.fromarray(cv2.cvtColor(cv2.resize(output, (self.disp_width, disp_height)), cv2.COLOR_BGR2RGB))
        imgtk_output = ImageTk.PhotoImage(image=output)
        self.label_output.imgtk = imgtk_output
        self.label_output.configure(image=imgtk_output)

        # load style
        style = Image.fromarray(cv2.cvtColor(style, cv2.COLOR_BGR2RGB))
        imgtk_style = ImageTk.PhotoImage(image=style)
        self.label_style.imgtk = imgtk_style
        self.label_style.configure(image=imgtk_style)

        # load style info
        self.text_artist.configure(text=self.st.get_style_info())

    def video_play(self):
        if self.is_video_stop == False:
            self.cam.set_frame()
            self.config_label()
            self.master.after(1, self.video_play)
        else:
            self.config_label()
            return

    def video_stop(self):
        self.is_video_stop = not self.is_video_stop

        if self.is_video_stop == False:
            self.btn_print["state"] = tk.DISABLED
            self.btn_save["state"] = tk.DISABLED
        else:
            self.btn_save["state"] = tk.NORMAL

        # remove all print files
        if os.path.exists("./print/print.png"):
            os.remove(r"./print/print.png")

    def capture(self):
        # e.g. capture_2021/01/29-12:34:56
        cv2.imwrite(
            "./capture" + "/" + "./capture_%s.png" % datetime.now().strftime("%Y/%m/%d-%H:%M:%S"), st.get_output(frame)
        )

    def save(self):
        output = self.st.get_output(frame)

        oh, ow, _ = output.shape
        disp_height = int(disp_width * (oh / ow))

        """
        here to start the merge output & style image
        """
        style_downscale = imutils.resize(st.get_style(), width=100, inter=cv2.INTER_AREA)
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

        # TODO: write date
        today = datetime.now().strftime("%Y-%m-%d")

        pass

    def print_out(self):
        os.system("lpr ./print/print.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", default=0, type=int, required=True, help="id of camera device")
    parser.add_argument("--width", default=700, type=int, required=True, help="width of output image")
    parser.add_argument("--disp_width", default=1215, type=int, required=True, help="width of window area")
    parser.add_argument("--num_sec", default=10, type=int, required=True, help="autoplay interval")
    parser.add_argument("--models", type=str, required=True, help="path/to/models.csv")
    return parser.parse_args()


def main():
    args = parse_args()  # parse arguments

    # TODO: check and create file dir (/print, /capture)

    root = tk.Tk()
    app = App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

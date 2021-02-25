import tensorflow as tf
import os
import utils
import style_transfer_tester
import argparse
import time
import csv

from os.path import isfile, join

"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--style_model", type=str, default="models/wave.ckpt", help="location for model file (*.ckpt)", required=True
    )

    parser.add_argument(
        "--content",
        type=str,
        default="content/female_knight.jpg",
        help="File path of content image (notation in the paper : x)",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="result.jpg",
        help="File path of output image (notation in the paper : y_c)",
        required=True,
    )

    parser.add_argument("--max_size", type=int, default=None, help="The maximum width or height of input images")

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --style_model
    try:
        # Tensorflow r0.12 requires 3 files related to *.ckpt
        assert (
            os.path.exists(args.style_model + ".index")
            and os.path.exists(args.style_model + ".meta")
            and os.path.exists(args.style_model + ".data-00000-of-00001")
        )
    except:
        print("There is no %s" % args.style_model)
        print("Tensorflow r0.12 requires 3 files related to *.ckpt")
        print("If you want to restore any models generated from old tensorflow versions, this assert might be ignored")
        return None

    # --content
    try:
        assert os.path.exists(args.content)
    except:
        print("There is no %s" % args.content)
        return None

    # --max_size
    try:
        if args.max_size is not None:
            assert args.max_size > 0
    except:
        print("The maximum width or height of input image must be positive")
        return None

    # --output
    dirname = os.path.dirname(args.output)
    try:
        if len(dirname) > 0:
            os.stat(dirname)
    except:
        os.mkdir(dirname)

    return args


"""main"""


def main():
    output_path = "output"
    content_path = "content"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open("final_model_list.csv", "r", encoding="utf-8") as f:
        models = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]

    contents = [join(content_path, f) for f in os.listdir(content_path) if isfile(join(content_path, f))]

    for content in contents:
        for model in models:
            tf.reset_default_graph()

            # load content image
            content_image = utils.load_image(content, max_size=None)

            # open session
            soft_config = tf.ConfigProto(allow_soft_placement=True)
            soft_config.gpu_options.allow_growth = True  # to deal with large image
            sess = tf.Session(config=soft_config)

            # build the graph
            transformer = style_transfer_tester.StyleTransferTester(
                session=sess, model_path=model["path"], content_image=content_image,
            )
            # execute the graph
            start_time = time.time()
            output_image = transformer.test()
            end_time = time.time()

            # save result
            output_path = "output/" + content.split("/")[1].split(".")[0] + "_" + model["path"].split("/")[1] + ".jpg"
            utils.save_image(output_image, output_path)

            # report execution time
            shape = content_image.shape  # (batch, width, height, channel)
            print(
                "===========Execution time for a %d x %d image : %f msec"
                % (shape[0], shape[1], 1000.0 * float(end_time - start_time) / 60)
            )

    # parse arguments
    # args = parse_args()
    # if args is None:
    #     exit()


if __name__ == "__main__":
    main()

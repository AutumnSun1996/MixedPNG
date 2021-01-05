import argparse
import numpy as np
import cv2.cv2 as cv

from utils import load_color as color, bgra2bgr, align
from gen_any import get_mixed, norm_img


def image_path(text):
    img = cv.imread(text)
    if img is None:
        raise ValueError("Invalid image %s" % text)
    return bgra2bgr(img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        "-i",
        nargs=2,
        metavar=("img1", "img2"),
        required=True,
        type=image_path,
    )
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument(
        "--background",
        "-b",
        nargs=2,
        metavar=("black", "white"),
        default=[color("black"), color("white")],
        type=color,
    )
    parser.add_argument("--gray", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--adjust-ratio", "-r", type=float, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    img1, img2 = align(*args.image)
    bg1, bg2 = args.background
    bg_template = np.ones_like(img1).astype("float")

    if args.reverse:
        img1, img2, bg1, bg2 = img2, img1, bg2, bg1

    res, _ = get_mixed(
        norm_img(img1),
        norm_img(img2),
        norm_img(bg1) * bg_template,
        norm_img(bg2) * bg_template,
        args.adjust_ratio,
    )
    cv.imwrite(args.output, res)


if __name__ == "__main__":
    main()

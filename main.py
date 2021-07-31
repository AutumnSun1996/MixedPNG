import os
import json
import argparse
import numpy as np
import cv2.cv2 as cv

from utils import load_color as color, bgra2bgr, align, get_image_data_url, pixel2str
from gen_any import get_mixed, norm_img


def image_path(text):
    img = cv.imread(text)
    if img is None:
        raise ValueError("Invalid image %s" % text)
    return bgra2bgr(img)


def bgr_to_gray(img):
    chl = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return np.stack([chl, chl, chl], -1)


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
    parser.add_argument(
        "--desaturate", "-ds", type=float, nargs="+", default=[0.5, 0.5]
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--gray", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--adjust-ratio", "-r", type=float, default=None)
    parser.add_argument("--trim-thresh", "-tr", type=float, default=1)
    args = parser.parse_args()
    return args


def generate_viewer(img1, img2, bg1, bg2, ced1, ced2, res, output):
    item = {
        "targets": [
            {"src": "img1.png"},
            {"src": "img2.png"},
            {"src": "ced1.png"},
            {"src": "ced2.png"},
        ],
        "backgrounds": [
            {"data": pixel2str(bg1)},
            {"data": pixel2str(bg2)},
        ],
        "result": {"src": 'result.png'},
    }
    with open("template/all.html", "r", -1, "UTF8") as f:
        template = f.read()
    result = template.replace("__JSON_DATA__", json.dumps(item))
    with open(f'{output}/index.html', "w", -1, "UTF8") as f:
        f.write(result)


def main():
    args = get_args()
    img1, img2 = align(*args.image)
    bg1, bg2 = args.background
    bg_template = np.ones_like(img1).astype("float")

    if args.reverse:
        img1, img2, bg1, bg2 = img2, img1, bg2, bg1

    if args.gray:
        img1 = bgr_to_gray(img1)
        img2 = bgr_to_gray(img2)

    img_name = args.output
    if not img_name.endswith('.png'):
        img_name += '.png'

    if args.verbose:
        tmp_dir = img_name[:-4] + '-step'
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = None

    res = get_mixed(
        norm_img(img1),
        norm_img(img2),
        norm_img(bg1) * bg_template,
        norm_img(bg2) * bg_template,
        args.desaturate,
        args.adjust_ratio,
        args.trim_thresh,
        tmp_dir,
    )

    cv.imwrite(img_name, res)


if __name__ == "__main__":
    main()

import base64
import os
import cv2.cv2 as cv
import numpy as np
from IPython.display import display, Image as IPyImage


def bgra2bgr(img, background=255):
    if img.shape[2] < 4:
        return img
    img = img.astype("float32")
    bgr = img[:, :, :3]
    alpha = img[:, :, 3:]
    result = bgr * (alpha / 255.0) + (255 - alpha) * (background / 255.0)
    return result.astype("uint8")


def div_no_zero(a, b, out=None):
    if out is None:
        # 当b为0时，默认输出0
        out = np.zeros_like(a)
    return np.divide(a, b, out=out, where=b != 0)


def img_dot(a, b):
    return np.einsum("hwd,hwd->hw", a, b)


def crop_center(img, w, h):
    """图像裁剪

    从中心截取大小为(w, h)的图像
    """
    h0, w0 = img.shape[:2]
    assert w0 >= w and h0 >= h, "Target size too large"
    if w == w0 and h == h0:
        return img
    dx = (w0 - w) // 2
    dy = (h0 - h) // 2
    return img[dy : dy + h, dx : dx + w]


def resize(img, w, h):
    """图像缩放

    保持比例缩放到刚好覆盖(w, h)
    """
    h0, w0 = img.shape[:2]
    scale_w = w / w0
    scale_h = h / h0
    if scale_w > scale_h:
        # 缩放以width为准
        h = int(h0 * scale_w)
    else:
        w = int(w0 * scale_h)
    print("Resize", w0, h0, w, h)
    return cv.resize(img, (w, h), interpolation=cv.INTER_CUBIC)


def align(a, b):
    """图像大小对齐

    返回a、b最小交集大小的图像，分别从中心截取
    """
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb)
    w = min(wa, wb)
    print("{}x{} and {}x{} => {}x{}".format(wa, ha, wb, hb, w, h))
    if wa > w and ha > h:
        a = resize(a, w, h)
    if wb > w and hb > h:
        b = resize(b, w, h)
    a = crop_center(a, w, h)
    b = crop_center(b, w, h)
    return a, b


def trim_matrix(mat, low, high, trim_low=None, trim_high=None):
    """矩阵数值裁剪和缩放

    :low: 结果的bgr最小值
    :high: 结果的bgr最小值
    :trim_low: 指定输入最小值，为None时将取输入矩阵最小值
    :trim_high: 指定输入最大值，为None时将取输入矩阵最小值
    """

    if trim_low is None:
        trim_low = mat.min()
    else:
        mat[mat < trim_low] = trim_low
    if trim_high is None:
        trim_high = mat.max()
    else:
        mat[mat > trim_high] = trim_high

    mat = low + (high - low) * (mat - trim_low) / (trim_high - trim_low)
    return mat


def showinfo(*args, **kwargs):
    pargs = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            arg = "<{}{}: {:.1f} | {:.1f} | {:.1f}>".format(
                arg.shape, arg.dtype, arg.min(), arg.mean(), arg.max()
            )
        pargs.append(arg)
    print(*pargs, **kwargs)


def show(img):
    if isinstance(img, bytes):
        img = IPyImage(img)
    elif isinstance(img, np.ndarray):
        if img.max() <= 1:
            img = img * 255
        data = cv.imencode(".png", img)[1].tobytes()
        img = IPyImage(data)
    elif isinstance(img, str) and os.path.isfile(img):
        with open(img, "rb") as f:
            img = IPyImage(f.read())
    display(img)


def get_image_data_url(img, ext=".png"):
    if ext == ".png":
        mime = "png"
    else:
        mime = "jpg"
    image = cv.imencode(ext, img)[1].tobytes()
    data = base64.b64encode(image).decode("ascii")
    return "data:img/{}; base64,{}".format(mime, data)


def pixel2str(pixel):
    b, g, r = pixel
    return "rgb({}, {}, {})".format(r, g, b)

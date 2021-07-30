import base64
import re
import os
import cv2.cv2 as cv
import numpy as np
from IPython.display import display, Image as IPyImage

COLOR_NAME_MAP = {
    'black': '#000000',
    'navy': '#000080',
    'darkblue': '#00008b',
    'mediumblue': '#0000cd',
    'blue': '#0000ff',
    'darkgreen': '#006400',
    'green': '#008000',
    'teal': '#008080',
    'darkcyan': '#008b8b',
    'deepskyblue': '#00bfff',
    'darkturquoise': '#00ced1',
    'mediumspringgreen': '#00fa9a',
    'lime': '#00ff00',
    'springgreen': '#00ff7f',
    'aqua': '#00ffff',
    'cyan': '#00ffff',
    'midnightblue': '#191970',
    'dodgerblue': '#1e90ff',
    'lightseagreen': '#20b2aa',
    'forestgreen': '#228b22',
    'seagreen': '#2e8b57',
    'darkslategray': '#2f4f4f',
    'darkslategrey': '#2f4f4f',
    'limegreen': '#32cd32',
    'mediumseagreen': '#3cb371',
    'turquoise': '#40e0d0',
    'royalblue': '#4169e1',
    'steelblue': '#4682b4',
    'darkslateblue': '#483d8b',
    'mediumturquoise': '#48d1cc',
    'indigo': '#4b0082',
    'darkolivegreen': '#556b2f',
    'cadetblue': '#5f9ea0',
    'cornflowerblue': '#6495ed',
    'rebeccapurple': '#663399',
    'mediumaquamarine': '#66cdaa',
    'dimgray': '#696969',
    'dimgrey': '#696969',
    'slateblue': '#6a5acd',
    'olivedrab': '#6b8e23',
    'slategray': '#708090',
    'slategrey': '#708090',
    'lightslategray': '#778899',
    'lightslategrey': '#778899',
    'mediumslateblue': '#7b68ee',
    'lawngreen': '#7cfc00',
    'chartreuse': '#7fff00',
    'aquamarine': '#7fffd4',
    'maroon': '#800000',
    'purple': '#800080',
    'olive': '#808000',
    'gray': '#808080',
    'grey': '#808080',
    'skyblue': '#87ceeb',
    'lightskyblue': '#87cefa',
    'blueviolet': '#8a2be2',
    'darkred': '#8b0000',
    'darkmagenta': '#8b008b',
    'saddlebrown': '#8b4513',
    'darkseagreen': '#8fbc8f',
    'lightgreen': '#90ee90',
    'mediumpurple': '#9370db',
    'darkviolet': '#9400d3',
    'palegreen': '#98fb98',
    'darkorchid': '#9932cc',
    'yellowgreen': '#9acd32',
    'sienna': '#a0522d',
    'brown': '#a52a2a',
    'darkgray': '#a9a9a9',
    'darkgrey': '#a9a9a9',
    'lightblue': '#add8e6',
    'greenyellow': '#adff2f',
    'paleturquoise': '#afeeee',
    'lightsteelblue': '#b0c4de',
    'powderblue': '#b0e0e6',
    'firebrick': '#b22222',
    'darkgoldenrod': '#b8860b',
    'mediumorchid': '#ba55d3',
    'rosybrown': '#bc8f8f',
    'darkkhaki': '#bdb76b',
    'silver': '#c0c0c0',
    'mediumvioletred': '#c71585',
    'indianred': '#cd5c5c',
    'peru': '#cd853f',
    'chocolate': '#d2691e',
    'tan': '#d2b48c',
    'lightgray': '#d3d3d3',
    'lightgrey': '#d3d3d3',
    'thistle': '#d8bfd8',
    'orchid': '#da70d6',
    'goldenrod': '#daa520',
    'palevioletred': '#db7093',
    'crimson': '#dc143c',
    'gainsboro': '#dcdcdc',
    'plum': '#dda0dd',
    'burlywood': '#deb887',
    'lightcyan': '#e0ffff',
    'lavender': '#e6e6fa',
    'darksalmon': '#e9967a',
    'violet': '#ee82ee',
    'palegoldenrod': '#eee8aa',
    'lightcoral': '#f08080',
    'khaki': '#f0e68c',
    'aliceblue': '#f0f8ff',
    'honeydew': '#f0fff0',
    'azure': '#f0ffff',
    'sandybrown': '#f4a460',
    'wheat': '#f5deb3',
    'beige': '#f5f5dc',
    'whitesmoke': '#f5f5f5',
    'mintcream': '#f5fffa',
    'ghostwhite': '#f8f8ff',
    'salmon': '#fa8072',
    'antiquewhite': '#faebd7',
    'linen': '#faf0e6',
    'lightgoldenrodyellow': '#fafad2',
    'oldlace': '#fdf5e6',
    'red': '#ff0000',
    'fuchsia': '#ff00ff',
    'magenta': '#ff00ff',
    'deeppink': '#ff1493',
    'orangered': '#ff4500',
    'tomato': '#ff6347',
    'hotpink': '#ff69b4',
    'coral': '#ff7f50',
    'darkorange': '#ff8c00',
    'lightsalmon': '#ffa07a',
    'orange': '#ffa500',
    'lightpink': '#ffb6c1',
    'pink': '#ffc0cb',
    'gold': '#ffd700',
    'peachpuff': '#ffdab9',
    'navajowhite': '#ffdead',
    'moccasin': '#ffe4b5',
    'bisque': '#ffe4c4',
    'mistyrose': '#ffe4e1',
    'blanchedalmond': '#ffebcd',
    'papayawhip': '#ffefd5',
    'lavenderblush': '#fff0f5',
    'seashell': '#fff5ee',
    'cornsilk': '#fff8dc',
    'lemonchiffon': '#fffacd',
    'floralwhite': '#fffaf0',
    'snow': '#fffafa',
    'yellow': '#ffff00',
    'lightyellow': '#ffffe0',
    'ivory': '#fffff0',
    'white': '#ffffff',
}


def bgra2bgr(img, background=255):
    """模拟background颜色的背景, 将bgra格式的图片转换为bgr格式
    """
    if img.shape[2] < 4:
        return img
    img = img.astype("float32")
    bgr = img[:, :, :3]
    alpha = img[:, :, 3:]
    result = bgr * (alpha / 255.0) + (255 - alpha) * (background / 255.0)
    return result.astype("uint8")


def div_no_zero(a, b, out=None):
    """进行矩阵除法, 但当b为0时将结果置为0
    """
    if out is None:
        # 当b为0时，默认输出0
        out = np.zeros_like(a)
    return np.divide(a, b, out=out, where=b != 0)


def img_dot(a, b):
    """对图片上的像素点进行点积
    """
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
    """模拟print, 但对numpy矩阵只输出元信息而不输出内容
    """
    pargs = []
    for arg in args:
        # 处理numpy矩阵
        if isinstance(arg, np.ndarray):
            arg = "<{}{}: min={:.3f} | avg={:.3f} | max={:.3f}>".format(
                arg.shape, arg.dtype, arg.min(), arg.mean(), arg.max()
            )
        pargs.append(arg)
    print(*pargs, **kwargs)


def show(img):
    """展示图片
    """
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
    """将numpy矩阵格式的图片转换为data url字符串
    """
    if ext == ".png":
        mime = "png"
    else:
        mime = "jpg"
    image = cv.imencode(ext, img)[1].tobytes()
    data = base64.b64encode(image).decode("ascii")
    return "data:img/{}; base64,{}".format(mime, data)


def pixel2str(pixel):
    """将像素值转换为css的rgb()字符串格式
    """
    b, g, r = pixel
    return "rgb({}, {}, {})".format(r, g, b)


def load_color(text):
    """将颜色字符串转化为(b,g,r)值
    """
    text = text.lower()
    if text in COLOR_NAME_MAP:
        text = COLOR_NAME_MAP[text]
    if re.search(r"^#[0-9a-f]{3}$", text):
        r, g, b = [int(text[i] * 2, 16) for i in (1, 2, 3)]
    elif re.search(r"^#[0-9a-f]{6}$", text):
        r, g, b = [int(text[i : i + 2], 16) for i in (1, 3, 5)]
    else:
        raise ValueError("Invalid Color Represent: %r" % text)
    return b, g, r

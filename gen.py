import cv2.cv2 as cv
import numpy as np


def showinfo(*args, **kwargs):
    pargs = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            arg = "<{}{}: {:.1f} | {:.1f} | {:.1f}>".format(
                arg.shape, arg.dtype, arg.min(), arg.mean(), arg.max()
            )
        pargs.append(arg)
    print(*pargs, **kwargs)


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


def preprocess(img, low, high, ignore_percentile_low=1, ignore_percentile_high=98):
    """图像亮度调整，按百分比舍去低亮度和高亮度部分，并重新进行scale

    :low: 结果的bgr最小值
    :high: 结果的bgr最小值
    :ignore_percentile_low: bgr在该百分比之下的将被设置为low
    :ignore_percentile_high: bgr在该百分比之上的将被设置为high
    """

    img = img.copy()
    colors = np.ravel(img)
    trim_low = np.percentile(colors, ignore_percentile_low)
    trim_high = np.percentile(colors, ignore_percentile_high)
    res = trim_matrix(img.astype("float"), low, high, trim_low, trim_high)
    return res


def desaturate(img, saturation_max):
    """图像去饱和

    :saturation_max: 结果的饱和度最大值
    """

    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS).astype("float")
    hls[:, :, 2] = trim_matrix(hls[:, :, 2], 0, saturation_max)
    bgr = cv.cvtColor(hls.astype("uint8"), cv.COLOR_HLS2BGR)
    return bgr


def div_no_zero(a, b, out=None):
    if out is None:
        # 当b为0时，默认输出0
        out = np.zeros_like(a)
    return np.divide(a, b, out=out, where=b != 0)


def gen_shadow_colored(
    white_image,
    black_image,
    light_bins=(0, 128, 128, 255),
    saturation_limit=128,
    ignore_percentile_low=10,
    ignore_percentile_high=95,
    black_first=True,
):
    """生成幻影图片

    :white_image: 白色背景时显示的图片
    :black_image: 黑色背景时显示的图片
    :light_bins: 重设亮度
    :saturation_limit: 重设最大饱和度
    :ignore_percentile_low: 重设亮度时将低于该比例的低亮度像素设置为最低亮度
        例如, 为10时将会使最暗的10%像素均被设置为最低亮度
    :ignore_percentile_high: 重设亮度时将高于该比例的高亮度像素设置为最高亮度
        例如, 为95时将会使最亮的5%像素均被设置为最高亮度
    :black_first: 优先保证黑色背景时的还原效果
        将导致另一背景下图像色彩较差
    """
    assert white_image.shape == black_image.shape, "图像大小应当一致"
    assert len(white_image.shape) == 3 and white_image.shape[2] == 3, "需要BGR格式图片"
    # 处理允许分别指定/同时指定的参数
    if isinstance(saturation_limit, (int, float)):
        saturation_limit = [saturation_limit, saturation_limit]
    if isinstance(ignore_percentile_low, (int, float)):
        ignore_percentile_low = [ignore_percentile_low, ignore_percentile_low]
    if isinstance(ignore_percentile_high, (int, float)):
        ignore_percentile_high = [ignore_percentile_high, ignore_percentile_high]
    # 黑色背景时显示的图像使用第一个参数
    desaturated_b = preprocess(
        desaturate(black_image, saturation_limit[0]),
        light_bins[0],
        light_bins[1],
        ignore_percentile_low[0],
        ignore_percentile_high[0],
    )
    showinfo("desaturated_b", desaturated_b)

    # 白色背景时显示的图像使用第二个参数
    desaturated_w = preprocess(
        desaturate(white_image, saturation_limit[1]),
        light_bins[2],
        light_bins[3],
        ignore_percentile_low[1],
        ignore_percentile_high[1],
    )
    showinfo("desaturated_w", desaturated_w)

    # 计算alpha通道
    alpha = (
        255.0 - desaturated_w.astype("float") + desaturated_b.astype("float")
    ).mean(axis=2, keepdims=True)
    showinfo("alpha", alpha)
    # 计算bgr通道
    if black_first:
        # 优先保证黑色背景下的显示效果
        result_bgr = div_no_zero(desaturated_b, alpha) * 255.0
    else:
        # 优先保证白色背景下的显示效果
        result_bgr = div_no_zero((desaturated_w - 255.0), alpha) * 255.0 + 255.0
    showinfo("result_bgr", result_bgr)
    # 保证bgr通道取值为0到255之间，溢出部分直接舍去
    # TODO: 是否可以避免溢出? 是否有更好的溢出处理方案?
    result_bgr = trim_matrix(result_bgr, 0, 255, 0, 255)
    showinfo("result_bgr", result_bgr)

    # 生成最终的图像
    mixed_image = np.concatenate((result_bgr, alpha), -1)
    showinfo("mixed_image", mixed_image)
    return mixed_image.astype("uint8")


def gen_shadow_gray(
    white_image,
    black_image,
    light_bins=(0, 128, 128, 255),
    ignore_percentile_low=10,
    ignore_percentile_high=95,
):
    """生成幻影图片

    :white_image: 白色背景时显示的图片
    :black_image: 黑色背景时显示的图片
    :light_bins: 重设亮度
    :ignore_percentile_low: 重设亮度时将低于该比例的低亮度像素设置为最低亮度
        例如, 为10时将会使最暗的10%像素均被设置为最低亮度
    :ignore_percentile_high: 重设亮度时将高于该比例的高亮度像素设置为最高亮度
        例如, 为95时将会使最亮的5%像素均被设置为最高亮度
    """
    assert white_image.shape == black_image.shape, "图像大小应当一致"
    assert len(white_image.shape) == 2, "需要灰度图"
    # 处理允许分别指定/同时指定的参数
    if isinstance(ignore_percentile_low, (int, float)):
        ignore_percentile_low = [ignore_percentile_low, ignore_percentile_low]
    if isinstance(ignore_percentile_high, (int, float)):
        ignore_percentile_high = [ignore_percentile_high, ignore_percentile_high]
    # 黑色背景时显示的图像使用第一个参数
    imgb = preprocess(
        imgb,
        light_bins[0],
        light_bins[1],
        ignore_percentile_low[0],
        ignore_percentile_high[0],
    )
    showinfo("imgb", imgb)

    # 白色背景时显示的图像使用第二个参数
    imgw = preprocess(
        imgw,
        light_bins[2],
        light_bins[3],
        ignore_percentile_low[1],
        ignore_percentile_high[1],
    )
    showinfo("imgw", imgw)

    # 计算alpha通道
    alpha = 255.0 - imgw.astype("float") + imgb.astype("float")
    showinfo("alpha", alpha)
    # 计算bgr通道
    gray = div_no_zero(imgb, alpha) * 255.0
    showinfo("gray", gray)
    # 保证bgr通道取值为0到255之间，溢出部分直接舍去
    # TODO: 是否可以避免溢出? 是否有更好的溢出处理方案?
    gray = trim_matrix(gray, 0, 255, 0, 255)
    showinfo("gray", gray)

    # 生成最终的图像
    mixed_image = np.stack((gray, gray, gray, alpha), -1)
    showinfo("mixed_image", mixed_image)
    return mixed_image.astype("uint8")


def main(path):
    img_white = cv.imread(path["white"])
    img_black = cv.imread(path["black"])
    cropw, cropb = align(img_white, img_black)
    res = gen_shadow_colored(cropw, cropb)
    cv.imwrite("gen.png", res)
    res = gen_shadow_colored(cropw, cropb, black_first=False)
    cv.imwrite("gen2.png", res)


if __name__ == "__main__":
    path = {
        "white": R"D:\Pictures\Wallpapers\735962.png",
        "black": R"D:\Pictures\Wallpapers\1500371699026.jpg",
    }
    main(path)

import cv2.cv2 as cv
import numpy as np

from utils import showinfo, img_dot, div_no_zero, trim_matrix, pixel2str, align, bgra2bgr


def rescale(img, factor):
    return factor[0] + img * factor[1]


def get_mixed(img1, img2, bg1=None, bg2=None, adjust_ratio=None):
    """生成幻影图片

    :param img1: 目标图片1, 在背景1下展示
    :param img2: 目标图片2, 在背景2下展示
    :param bg1: 背景1, 默认为全白色
    :param bg2: 背景2, 默认为全黑色
    :param adjust_ratio: float, 0到1之间, 默认根据背景确定为0或1

    所有图像(包括背景)均应满足:
        shape为(h,w,3)或(w,h,3)
        float格式
        [0, 1]区间
    """
    showinfo("get_mixed", img1, img2, bg1, bg2)
    # 计算背景颜色差异
    diff_bg = bg1 - bg2
    diff_bg_2 = img_dot(diff_bg, diff_bg)

    # 根据背景进行颜色调制
    # ==色相调制==
    if diff_bg.mean() > 0:
        print("背景颜色调制 1")
        img1 = img1 * diff_bg + bg2
        img2 = img2 * diff_bg + bg2
        default_adjust = 1
    else:
        print("背景颜色调制 -1")
        img1 = bg1 - img1 * diff_bg
        img2 = bg1 - img2 * diff_bg
        default_adjust = 0
    ced = [img1.copy() * 255, img2.copy() * 255]

    # ==亮度差调制==
    if adjust_ratio is None:
        adjust_ratio = default_adjust

    # 计算目标颜色差异
    diff_img = img1 - img2

    # 先尝试直接计算透明度
    beta = div_no_zero(img_dot(diff_bg, diff_img), diff_bg_2)
    alpha = 1 - beta
    showinfo("init alpha", alpha)

    # 获取透明度极值点
    min_pos = np.unravel_index(np.argmin(beta), beta.shape)
    max_pos = np.unravel_index(np.argmax(beta), beta.shape)
    print(
        "beta: min=%.3f@%s, max=%.3f@%s"
        % (beta[min_pos], min_pos, beta[max_pos], max_pos)
    )
    print(
        "init alpha: min=%.3f@%s, max=%.3f@%s"
        % (
            alpha[max_pos],
            max_pos,
            alpha[min_pos],
            min_pos,
        )
    )

    # 当极值满足限制时，无需修改
    if beta[min_pos] >= 0 and beta[max_pos] <= 1:
        print("No change Required")
        fimg1 = img1
        # fimg2 = img2
    else:
        # 计算新的极值
        new_max = min(beta[max_pos], 1)
        new_min = max(beta[min_pos], 0)
        print("new range", new_min, new_max)

        # 取当前极值点数值
        beta_max = beta[max_pos]
        beta_min = beta[min_pos]

        r_max = diff_bg[max_pos].sum() / np.dot(diff_bg[max_pos], diff_bg[max_pos])
        r_min = diff_bg[min_pos].sum() / np.dot(diff_bg[min_pos], diff_bg[min_pos])

        # 计算变换参数
        diff_b = (new_min * r_max - new_max * r_min) / (
            beta_min * r_max - beta_max * r_min
        )
        diff_a = -(beta_max * diff_b - new_max) / r_max

        print("a", diff_a, "b", diff_b)

        # 计算变换参数 a1 的取值范围
        max_a1 = min(1 - diff_b * img1.max(), 1 + diff_a - diff_b * img2.max())
        min_a1 = max(diff_a - diff_b * img2.min(), -diff_b * img1.min())
        print("adjust limit:", min_a1, max_a1, "ratio:", adjust_ratio)

        # 得到图像1和2的变换参数
        factor1 = min_a1 + adjust_ratio * (max_a1 - min_a1), diff_b
        factor2 = factor1[0] - diff_a, diff_b

        # 进行变换
        fimg1 = rescale(img1, factor1)
        showinfo("rescale1 = %.4f + img * %.4f:" % factor1, fimg1)
        fimg2 = rescale(img2, factor2)
        showinfo("rescale2 = %.4f + img * %.4f:" % factor2, fimg2)
        # 更新图像差异
        diff_img = fimg1 - fimg2
        # 更新透明度
        alpha = 1 - div_no_zero(img_dot(diff_bg, diff_img), diff_bg_2)

    showinfo("alpha final", alpha)
    print(
        "new alpha: min=%.3f@%s, max=%.3f@%s"
        % (
            alpha[max_pos],
            max_pos,
            alpha[min_pos],
            min_pos,
        )
    )
    # 将透明度矩阵扩展为(h,w,1)
    alpha = np.expand_dims(alpha, -1)

    # 计算bgr通道数值
    result_bgr = bg1 + div_no_zero(fimg1 - bg1, alpha)
    showinfo("result_bgr ", result_bgr)
    # 保证bgr通道取值为0到1之间，溢出部分直接舍去
    # TODO: 是否可以避免溢出? 是否有更好的溢出处理方案?
    # TODO: 饱和度调制?
    result_bgr = trim_matrix(result_bgr, 0, 1, 0, 1)
    showinfo("trimmed_bgr", result_bgr)

    # 生成最终的图像
    mixed_image = np.concatenate((result_bgr, alpha), -1)
    showinfo("mixed_image", mixed_image)
    # TODO: 是否需要 * 255? 是否需要转换为 uint8?
    return mixed_image * 255, ced


def norm_img(img):
    return np.array(img).astype("float") / 255


def generate_html(img1, img2, bg1, bg2, name="all", adjust_ratio=None):
    from utils import get_image_data_url
    import json

    bg_template = np.ones_like(img1).astype("float")
    res, (ced1, ced2) = get_mixed(
        norm_img(img1),
        norm_img(img2),
        norm_img(bg1) * bg_template,
        norm_img(bg2) * bg_template,
        adjust_ratio,
    )
    cv.imwrite("output/%s.png" % name, res)

    item = {
        "targets": [
            {"data": get_image_data_url(img1, ".jpg")},
            {"data": get_image_data_url(img2, ".jpg")},
            {"data": get_image_data_url(ced1, ".jpg")},
            {"data": get_image_data_url(ced2, ".jpg")},
        ],
        "backgrounds": [
            {"data": pixel2str(bg1)},
            {"data": pixel2str(bg2)},
        ],
        "result": {"data": "../output/%s.png" % name},
    }
    with open("template/all.html", "r", -1, "UTF8") as f:
        template = f.read()
    result = template.replace("__JSON_DATA__", json.dumps(item))
    with open("output/%s.html" % name, "w", -1, "UTF8") as f:
        f.write(result)


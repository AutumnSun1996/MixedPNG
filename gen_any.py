import cv2.cv2 as cv
import numpy as np

from utils import (
    showinfo,
    img_dot,
    div_no_zero,
    trim_matrix,
    desaturate,
)


def rescale(img, a, b):
    return a + img * b


def get_mixed(
    img1,
    img2,
    bg1=None,
    bg2=None,
    s_max=None,
    adjust_ratio=None,
    trim_thresh=1,
    tmp_img_dir=None,
):
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
    showinfo("背景", bg1, bg2)
    showinfo("输入", img1, img2)
    # 计算背景颜色差异
    diff_bg = bg1 - bg2
    diff_bg_2 = img_dot(diff_bg, diff_bg)
    if tmp_img_dir:
        cv.imwrite(f'{tmp_img_dir}/0-src-1.png', img1 * 255)
        cv.imwrite(f'{tmp_img_dir}/0-src-2.png', img2 * 255)
        cv.imwrite(f'{tmp_img_dir}/0-bg-1.png', bg1 * 255)
        cv.imwrite(f'{tmp_img_dir}/0-bg-2.png', bg2 * 255)

    # ==去饱和==
    if s_max is not None:
        img1 = desaturate(img1, s_max[0])
        img2 = desaturate(img2, s_max[1])
    if tmp_img_dir:
        cv.imwrite(f'{tmp_img_dir}/1-desaturate-1.png', img1 * 255)
        cv.imwrite(f'{tmp_img_dir}/1-desaturate-2.png', img2 * 255)
    showinfo('去饱和', img1, img2)

    # 根据背景进行颜色调制
    # ==色相调制==
    if diff_bg.mean() > 0:
        # bg1平均亮度更高
        img1 = img1 * diff_bg + bg2
        img2 = img2 * diff_bg + bg2
        default_adjust = 1
    else:
        # bg2平均亮度更高
        img1 = bg1 - img1 * diff_bg
        img2 = bg1 - img2 * diff_bg
        default_adjust = 0
    if tmp_img_dir:
        cv.imwrite(f'{tmp_img_dir}/2-bg-color-1.png', img1 * 255)
        cv.imwrite(f'{tmp_img_dir}/2-bg-color-2.png', img2 * 255)
    showinfo('色相调制', img1, img2)

    # ==亮度差调制参数==
    if adjust_ratio is None:
        adjust_ratio = 0.5

    # 计算目标颜色差异
    diff_img = img1 - img2

    # 先尝试直接计算透明度
    beta = div_no_zero(img_dot(diff_bg, diff_img), diff_bg_2)
    # 获取透明度极值点
    min_pos = np.unravel_index(np.argmin(beta), beta.shape)
    max_pos = np.unravel_index(np.argmax(beta), beta.shape)
    showinfo("init beta", beta)

    # 当极值满足限制时，无需修改
    if beta[min_pos] >= 0 and beta[max_pos] <= 1:
        print("无需进行亮度调整")
    else:
        # 计算新的极值
        new_max = min(beta[max_pos], 1)
        new_min = max(beta[min_pos], 0)
        print("进行亮度调整", new_min, new_max)

        # 取当前极值点数值
        beta_max = beta[max_pos]
        beta_min = beta[min_pos]

        r_max = diff_bg[max_pos].sum() / np.dot(diff_bg[max_pos], diff_bg[max_pos])
        r_min = diff_bg[min_pos].sum() / np.dot(diff_bg[min_pos], diff_bg[min_pos])

        # 计算变换参数
        # diff_b = (r_min * (new_max - 1) - r_max * (new_min - 1)) / (
        #     beta_max * r_min - beta_min * r_max
        # )
        # diff_a = (1 - new_max + diff_b * beta_max) / r_max

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
        a1 = min_a1 + adjust_ratio * (max_a1 - min_a1)
        a2 = a1 - diff_a

        # 进行变换
        img1 = rescale(img1, a1, diff_b)
        showinfo("rescale1 = %.4f + img * %.4f:" % (a1, diff_b), img1)
        img2 = rescale(img2, a2, diff_b)
        showinfo("rescale2 = %.4f + img * %.4f:" % (a2, diff_b), img2)
        if tmp_img_dir:
            cv.imwrite(f'{tmp_img_dir}/2-lightness-1.png', img1 * 255)
            cv.imwrite(f'{tmp_img_dir}/2-lightness-2.png', img2 * 255)

        # 更新图像差异
        diff_img = img1 - img2
        # 更新透明度
        beta = div_no_zero(img_dot(diff_bg, diff_img), diff_bg_2)
        showinfo("new beta", beta)

    alpha = 1 - beta
    showinfo("final alpha", alpha)
    if tmp_img_dir:
        cv.imwrite(f'{tmp_img_dir}/4-result-alpha.png', alpha * 255)
    # 将透明度矩阵扩展为(h,w,1)
    alpha = np.expand_dims(alpha, -1)

    # 计算bgr通道数值
    result_bgr = bg1 + div_no_zero(img1 - bg1, alpha)
    showinfo("result_bgr ", result_bgr)
    # 保证bgr通道取值为0到1之间，溢出部分直接舍去
    # TODO: 是否可以避免溢出? 是否有更好的溢出处理方案?
    # TODO: 饱和度调制?
    trim_thresh = min(result_bgr.max(), 1, trim_thresh)
    result_bgr = trim_matrix(result_bgr, 0, 1, 0, trim_thresh)
    showinfo("trimmed_bgr", result_bgr)

    # 生成最终的图像
    mixed_image = np.concatenate((result_bgr, alpha), -1)
    if tmp_img_dir:
        cv.imwrite(f'{tmp_img_dir}/4-result-bgr.png', result_bgr * 255)
        cv.imwrite(f'{tmp_img_dir}/5-result.png', mixed_image * 255)
        cv.imwrite(
            f'{tmp_img_dir}/6-result-bg1.png',
            (result_bgr * alpha + bg1 * (1 - alpha)) * 255,
        )
        cv.imwrite(
            f'{tmp_img_dir}/6-result-bg2.png',
            (result_bgr * alpha + bg2 * (1 - alpha)) * 255,
        )
    showinfo("mixed_image", mixed_image)
    # TODO: 是否需要 * 255? 是否需要转换为 uint8?
    return mixed_image * 255


def norm_img(img):
    return np.array(img).astype("float") / 255

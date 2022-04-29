# 混合PNG生成

功能:

将两张图片合成为一张新的带有透明通道的新图片, 令新图片在指定的两种背景下的显示效果分别接近两张输入图片.

HTML Demo: [gh-pages](https://autumnsun1996.github.io/MixedPNG/)

TODO:
- 交互和UI优化
- [ ] 输入图像纵横比不同时进行自动裁剪, 并支持在html页面中手动进行裁剪
- [ ] 优化输出图像大小设置逻辑(当前为继承第一张图片, 考虑改为继承较大的图片并支持手动修改)
- [ ] 展示方式优化

- 图像处理算法优化
- [ ] 支持指定背景色
- [ ] 优化图像处理逻辑, 有重合像素时减少像素值缩放比例

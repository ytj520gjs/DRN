
import cv2
import numpy as np


def equalize_hist_16bit(img):
    """16位直方图均衡化"""
    hist, bins = np.histogram(img.flatten(), bins=65536, range=(0, 65535))
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 65535 / (cdf.max() - cdf.min())
    return np.interp(img.flatten(), bins[:-1], cdf_normalized).reshape(img.shape).astype(np.uint16)


def process_depth_map(file_path, use_16bit_enhancement=True):
    """处理深度图的主函数"""
    # 读取深度图
    depth_map = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    if depth_map is None:
        raise ValueError(f"无法读取文件: {file_path}")

    print(f"深度图信息 - 数据类型: {depth_map.dtype}, 最小值: {np.min(depth_map)}, 最大值: {np.max(depth_map)}")

    # 根据数据类型处理
    if depth_map.dtype == np.uint16:
        if use_16bit_enhancement:
            # 16位增强处理
            equalized = equalize_hist_16bit(depth_map)
            normalized_8bit = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            process_name = "16-bit Equalized"
        else:
            # 降级到8位处理
            normalized_8bit = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            normalized_8bit = cv2.equalizeHist(normalized_8bit)
            process_name = "8-bit Equalized"
    else:
        # 其他数据类型处理
        normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        normalized_8bit = normalized.astype(np.uint8)
        if normalized_8bit.dtype == np.uint8:
            normalized_8bit = cv2.equalizeHist(normalized_8bit)
        process_name = "8-bit Processed"

    # 伪彩色映射和显示（所有路径共用）
    colored = cv2.applyColorMap(normalized_8bit, cv2.COLORMAP_JET)

    cv2.imshow("Original Depth", cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    cv2.imshow(process_name, normalized_8bit)
    cv2.imshow("Color Mapped", colored)

    output_path = file_path.replace(".png", "_enhanced.png")
    cv2.imwrite(output_path, colored)
    print(f"增强结果已保存至: {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
file_path = r"F:\paper\read-papers\datasets\DRN-modify\OnlineChallenge\DepthImages\Depth_1.png"
process_depth_map(file_path, use_16bit_enhancement=True)
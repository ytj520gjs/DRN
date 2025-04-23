import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
# from time import time
import time
from tqdm import tqdm


def farthest_point_sampling(points, num_samples):
    """
    优化的最远点采样算法（带进度显示）
    :param points: 点云坐标数组(N, 3)
    :param num_samples: 目标采样点数
    :return: 采样点索引数组
    """
    n_points = points.shape[0]
    if num_samples > n_points:
        print(f"警告：采样数{num_samples}超过点云总数{n_points}，将返回完整点云")
        return np.arange(n_points)

    # 使用矩阵运算加速距离计算
    selected_indices = []
    remaining_indices = np.arange(n_points)

    # 随机选择初始点
    np.random.seed(42)  # 固定随机种子保证可重复性
    start_idx = np.random.randint(n_points)
    selected_indices.append(start_idx)
    remaining_indices = np.delete(remaining_indices, np.where(remaining_indices == start_idx)[0])

    # 预计算距离矩阵（内存足够时）
    if n_points < 50000:  # 根据内存调整阈值
        all_dists = np.linalg.norm(
            points[:, np.newaxis] - points,
            axis=2
        )

    print(f"开始采样 {num_samples} 个点（原始点云 {n_points} 点）")
    # start_time = time()导入方式变了，这里就改了
    start_time = time.time()

    for i in range(1, num_samples):
        if i % 100 == 0:
            # print(f"进度: {i}/{num_samples} | 耗时: {time() - start_time:.1f}s")
            print(f"进度: {i}/{num_samples} | 耗时: {time.time() - start_time:.1f}s")

        if 'all_dists' in locals():
            # 使用预计算的距离矩阵
            min_dists = np.min(all_dists[remaining_indices][:, selected_indices], axis=1)
        else:
            # 动态计算距离
            min_dists = np.min(np.linalg.norm(
                points[remaining_indices][:, np.newaxis] - points[selected_indices],
                axis=2
            ), axis=1)

        farthest_idx = np.argmax(min_dists)
        selected_indices.append(remaining_indices[farthest_idx])
        remaining_indices = np.delete(remaining_indices, farthest_idx)

    # print(f"采样完成！总耗时: {time() - start_time:.2f}秒")
    print(f"采样完成！总耗时: {time.time() - start_time:.2f}秒")
    return selected_indices


def process_and_visualize(input_path, output_path, target_points):
    """处理单个文件并可视化（保留颜色）"""
    # 读取点云
    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_points():
        print(f"警告: {input_path} 是空点云")
        return False

    original_points = np.asarray(pcd.points)
    print(f"\n处理文件: {input_path}")
    print(f"原始点云包含 {len(original_points)} 个点")

    # 提取颜色信息（如果有）
    has_colors = pcd.has_colors()
    if has_colors:
        original_colors = np.asarray(pcd.colors)
    else:
        original_colors = None

    # 执行采样
    sampled_indices = farthest_point_sampling(original_points, target_points)
    sampled_points = original_points[sampled_indices]

    # 提取对应采样点的颜色
    if has_colors:
        sampled_colors = original_colors[sampled_indices]

    # 保存采样结果
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

    if has_colors:
        sampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)

    o3d.io.write_point_cloud(output_path, sampled_pcd)
    print(f"已保存采样结果到 {output_path}（{len(sampled_points)}个点）")

    # 可视化
    # visualize_comparison(original_points, sampled_points)

    return True


# def visualize_comparison(original, sampled):
    """可视化对比函数"""
    # fig = plt.figure(figsize=(18, 8))
    #
    # 原始点云
    # ax1 = fig.add_subplot(121, projection='3d')
    # sc1 = ax1.scatter(
    #     original[:, 0], original[:, 1], original[:, 2],
        # s=1, c='blue', alpha=0.5, label=f'原始 ({len(original)}点)'
    # )
    # ax1.set_title('Original Point Cloud')
    # ax1.legend()

    # 采样点云
    # ax2 = fig.add_subplot(122, projection='3d')
    # sc2 = ax2.scatter(
        # sampled[:, 0], sampled[:, 1], sampled[:, 2],
        # s=10, c='red', alpha=1, label=f'采样 ({len(sampled)}点)'
    # )
    # ax2.set_title('FPS Sampled Point Cloud')
    # ax2.legend()

    # plt.tight_layout()
    # plt.show()


# def non_blocking_visualization(pcd, window_title, delay=1.5):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=window_title, width=800, height=600, visible=True)
#     vis.add_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(delay)
#     vis.destroy_window()

# 配置路径和参数
input_dir = r"F:\paper\read-papers\datasets\DRN-modify\OnlineChallenge\plyfiles_world"
output_dir = r"F:\paper\read-papers\datasets\DRN-modify\OnlineChallenge\sampled_plys"
target_points = 1024  # 目标采样点数
file_range = (1, 10)  # 处理ply_1.ply到ply_10.ply

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
start_total = time.time()
failed_files = []

# 批量处理文件+主循环带进度条
for i in tqdm(range(file_range[0], file_range[1] + 1),desc="处理进度"):
    input_file = os.path.join(input_dir, f"ply_{i}.ply")
    output_file = os.path.join(output_dir, f"ply_{i}_sampled.ply")

    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        continue

    # process_and_visualize(input_file, output_file, target_points)
    success = process_and_visualize(input_file, output_file, target_points)
    if not success:
        print(f"⚠️  跳过空点云或处理失败: {input_file}")
        continue
    # # Open3D可视化
    # print("\nOpen3D可视化控制指南：")
    # print("1. 鼠标左键旋转视角")
    # print("2. 鼠标滚轮缩放")
    # print("3. 按 'H' 键显示帮助菜单")
    # print("4. 按 'Q' 键退出查看器")

    # 非阻塞可视化
    # sampled_pcd = o3d.io.read_point_cloud(output_file)
    #
    # if sampled_pcd.has_points():
    #     non_blocking_visualization(sampled_pcd, f"ply_{i}_sampled.ply")
    # else:
    #     print(f"❌ 警告: 采样结果为空，跳过可视化: {output_file}")
    #     failed_files.append(output_file)
    #     continue
    #
    # o3d.visualization.draw_geometries(
    #     [sampled_pcd],
    #     window_name=f"采样结果 {len(np.asarray(sampled_pcd.points))}/{len(np.asarray(o3d.io.read_point_cloud(input_file).points))} 点",
    #     width=800,
    #     height=600
    # )
print(f"\n✅ 所有文件处理完成！总耗时: {time.time() - start_total:.2f} 秒")
if failed_files:
    print("\n⚠️ 以下文件处理失败或为空：")
    for f in failed_files:
        print(f" - {f}")
print("所有文件处理完成！")

#下面的先不用生成
#拼接多个采样图并保存成图像
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#import argparse引入命令行参数模块

# 放在你的主逻辑开始前（比如 input_dir = ... 上面）：
#

# parser = argparse.ArgumentParser(description="批量点云采样与可视化")
# parser.add_argument('--summary', action='store_true', help="是否保存所有采样结果拼图")
# args = parser.parse_args()
# 这段代码会自动识别是否传入 --summary，结果保存在 args.summary（True/False）。

#在你的主处理循环中加：
# sampled_pcd_list = []
# titles = []
# 然后在每次采样成功后添加：
# if sampled_pcd.has_points():
#     sampled_pcd_list.append(sampled_pcd)
#     titles.append(f"ply_{i}")

# def save_sampling_summary(pcd_list, titles, save_path="output_summary.png", cols=5):
#     """
#     将多个点云结果拼接可视化并保存为图像
#     :param pcd_list: List of open3d.geometry.PointCloud
#     :param titles: List of title strings
#     :param save_path: 输出图片路径
#     :param cols: 每行显示几个图
#     """
#     rows = (len(pcd_list) + cols - 1) // cols
#     fig = plt.figure(figsize=(4 * cols, 4 * rows))
#
#     for i, pcd in enumerate(pcd_list):
#         pts = np.asarray(pcd.points)
#         colors = np.asarray(pcd.colors)
#
#         ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
#         ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=colors if colors.any() else 'gray')
#         ax.set_title(titles[i], fontsize=10)
#         ax.axis('off')
#
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     print(f"📸 采样图已保存到: {save_path}")
#在所有点云处理完成之后加：
# if sampled_pcd_list:
#     save_sampling_summary(sampled_pcd_list, titles, save_path="F:/paper/sampling_summary.png")
# 将上面这部分替换成线面两行（下面两行包含参数设置的情况）
# if args.summary and sampled_pcd_list:
#     save_sampling_summary(sampled_pcd_list, titles, save_path="F:/paper/sampling_summary.png")
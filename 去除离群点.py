# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 1. 读取点云数据（请替换为你的点云文件路径）
# pcd = o3d.io.read_point_cloud("your_point_cloud.ply")
# points = np.asarray(pcd.points)
# print("点云中点的数量:", len(points))
#
# # 2. 设置最近邻参数：这里使用 k 个邻居（注意：查询时包含自身，所以传入 k+1）
# k = 20
#
# # 3. 建立KD-Tree
# kdtree = o3d.geometry.KDTreeFlann(pcd)
#
# # 4. 对每个点计算其 k 个最近邻（排除自身）的平均距离
# avg_distances = []
# for point in points:
# # 查询 k+1 个最近邻，第一个返回的点是自己，其距离为 0
#     [_, idx, dists] = kdtree.search_knn_vector_3d(point, k + 1)
#     # dists 是平方距离，排除第一个(自己)，然后取平方根得到实际距离
#     dists = np.sqrt(np.asarray(dists)[1:])  # 排除自身的距离
#     avg_distance = np.mean(dists)
#     avg_distances.append(avg_distance)
#
#     avg_distances = np.array(avg_distances)
#
# # 5. 绘制直方图观察平均邻域距离的分布
# plt.figure(figsize=(8, 5))
# plt.hist(avg_distances, bins=50, color='blue', alpha=0.7)
# plt.title("每个点的平均邻域距离直方图")
# plt.xlabel("平均距离")
# plt.ylabel("频数")
# plt.show()


import numpy as np
import open3d as o3d

# 读取点云（请替换为你的点云文件路径）
file_path = "F:/paper/read-papers/datasets/DRN-modify/OnlineChallenge/plyfiles_world/ply_1.ply"
pcd = o3d.io.read_point_cloud(file_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 方法1：全局统计，获取全局最小值和最大值
global_min = points.min(axis=0)
global_max = points.max(axis=0)
print("全局边界：", global_min, global_max)

# 手动设置ROI边界（这里假设彩色区域在x轴正方向上，且有一定范围）
# 注意：这些值需要根据实际情况进行调整
roi_min = np.array([global_min[0] + (global_max[0] - global_min[0]) * 0.5,  # x轴中点偏右
                    global_min[1],  # y轴最小值
                    global_min[2]])  # z轴最小值
roi_max = np.array([global_max[0],  # x轴最大值
                    global_max[1],  # y轴最大值
                    global_max[2] * 0.75])  # z轴稍微小于最大值，假设彩色区域不在最高点
# # 假设目标区域大致位于全局边界的中心附近
# center = (global_min + global_max) / 2.0
# # 根据先验信息设定目标区域的尺寸（例如1m×1m×1m）
# roi_size = np.array([1.0, 1.0, 1.0])
# min_bound = center - roi_size / 2.0
# max_bound = center + roi_size / 2.0

# print("ROI 边界：", min_bound, max_bound)
print("ROI 边界：", roi_min, roi_max)
# 利用 Open3D 的 AxisAlignedBoundingBox 对 ROI 进行可视化
# bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=roi_min, max_bound=roi_max)
bbox.color = (1, 0, 0)  # 红色显示边界

# 可视化点云和 ROI 边界框
o3d.visualization.draw_geometries([pcd, bbox])

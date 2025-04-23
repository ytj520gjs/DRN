import os
import numpy as np
import cv2
from path import Path
from tqdm import tqdm


# 构造 K 矩阵
K = np.array([
    [1371.58264160156, 0.0, 973.902038574219],
    [0.0, 1369.42761230469, 537.702270507812],
    [0.0, 0.0, 1.0]
])

# 将 K 矩阵展开成一维数组并存储到文件
K.flatten().tofile("K.txt", sep="\n")
print("K.txt 文件已成功生成")



# 构造 pose 矩阵
rotation = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

translation = np.array([[0.0], [0.0], [0.0]])

# 构造 4x4 外参矩阵 pose
R = rotation.reshape(3, 3)  # 将旋转矩阵从一维数组转为 3x3
T = translation.reshape(3, 1)  # 将平移向量转为列向量

pose = np.hstack((R, T))  # 横向拼接旋转矩阵和平移向量
pose = np.vstack((pose, np.array([0.0, 0.0, 0.0, 1.0])))  # 添加最后一行 [0 0 0 1]
# 保存为 txt 文件
pose.flatten().tofile("poses.txt", sep="\n")
print("poses.txt 文件已成功生成")


def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d       
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red     
    property uchar alpha   
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()


def depth_image_to_point_cloud(rgb, depth, scale, K, pose):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])  #表示图像的宽度[1]和高度[0]即表示图像的像素点u（x），v（y）

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)#扁平化：将二维数据转化成一维，方便结合坐标和颜色信息
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]#筛选无效点减少点云中的噪声

    position = np.vstack((X, Y, Z, np.ones(len(X))))#齐次坐标系转换：np.vstack：将数组按行堆叠
    position = np.dot(pose, position)#相机坐标转世界坐标

    R = np.ravel(rgb[:, :, 0])[valid]#提取红色通道的像素值（高度（行），宽度（列））
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()
#np.transpose() 转置矩阵，将形状从 (6, N) 变为 (N, 6)，np.vstack() 用于将多个一维或二维数组按 垂直方向 组合。转换成列表的形式
    return points
# dataset_path = "F:/paper/read-papers/datasets/DRN-modify/DRN/OnlineChallenge"
#
# print("K.txt 路径:", os.path.join(dataset_path, "K.txt"))
# print("K.txt 是否存在:", os.path.exists(os.path.join(dataset_path, "K.txt")))

# image_files: XXXXXX.png (RGB, 24-bit, PNG)
# depth_files: XXXXXX.png (16-bit, PNG)
# poses: camera-to-world, 4×4 matrix in homogeneous coordinates
def build_point_cloud(dataset_path, scale, view_ply_in_world_coordinate):
    K = np.fromfile(os.path.join(dataset_path, "K.txt"), dtype=float, sep="\n ")
    K = np.reshape(K, newshape=(3, 3))
    rgb_path = os.path.dirname(dataset_path) + '/OnlineChallenge/RGBImages'
    depth_path = os.path.dirname(dataset_path) + '/OnlineChallenge/DepthImages'
    rgb_files = sorted(Path(rgb_path).files('*.png'))
    depth_files = sorted(Path(depth_path).files('*.png'))

    if view_ply_in_world_coordinate:
        poses = np.fromfile(os.path.join(dataset_path, "poses.txt"), dtype=float, sep="\n ")
        poses = np.reshape(poses, newshape=(-1, 4, 4))#这里对pose维度进行了定义
    else:
        poses = np.eye(4)#单位矩阵
    print("Poses shape:", poses.shape)
    for i in tqdm(range(1, 151)):#循环处理每一帧的图像
        image_file = rgb_files[i-1]
        depth_file = depth_files[i-1]

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)

        if view_ply_in_world_coordinate:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[0])#输出poses中的第1个矩阵
            # current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[(i-1) % 3])#这里%3是因为原来的poses.txt中有三个矩阵保证在整个过程中位姿矩阵的索引可以循环使用这 3 个矩阵
            #pose=poses[i%3]从 poses 数组中循环取位姿矩阵，若 poses 仅包含 3 个矩阵，则 i % 3 确保在迭代中循环使用它们。
            # current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[i])
        else:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses)
        save_ply_name = f'ply_{i}.ply'
        save_ply_path = os.path.dirname(dataset_path) + '/OnlineChallenge/plyfiles_world'

        if not os.path.exists(save_ply_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(save_ply_path)
        write_point_cloud(os.path.join(save_ply_path, save_ply_name), current_points_3D)

print("K 矩阵为:\n", K)
print("poses 矩阵为:\n", pose)
def generate_ply_file(image_path):
    # 如果view_ply_in_world_coordinate为True,那么点云的坐标就是在world坐标系下的坐标，否则就是在当前帧下的坐标
    view_ply_in_world_coordinate = True
    # 深度图对应的尺度因子，即深度图中存储的值与真实深度（单位为m）的比例, depth_map_value / real depth = scale_factor
    # 不同数据集对应的尺度因子不同，比如TUM的scale_factor为5000， hololens的数据的scale_factor为1000, Apollo Scape数据的scale_factor为200
    scale_factor = 1000.0
    build_point_cloud(image_path, scale_factor, view_ply_in_world_coordinate)


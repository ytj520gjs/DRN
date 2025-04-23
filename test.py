import json
import numpy as np
import os
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt


from sklearn.metrics import mean_squared_error, r2_score
from model import Net_LSTM  # 请替换成你真正的模型类名
from torch_geometric.data import Data
import argparse

# -----------------------
# 模型、数据相关函数
# -----------------------

# ========== 加载模型 ==========
def load_model(model_path, device):
    model = Net_LSTM(out_features=1)#初始化本实验模型
    model.load_state_dict(torch.load(model_path, map_location=device))#加载训练好的模型
    model.to(device)
    model.eval()#切换评估模式
    return model

# 加载 GroundTruth JSON 文件
def json_load(filename):
    with open(filename, "r") as fr:
        vars = json.load(fr)
    # for k, v in vars.items():
    #     vars[k] = np.array(v)
    return vars#vars=data可换为return data['Measurements']

# # 读取 GroundTruth 所有标签
def read_labels(json_path):
    with open(json_path, 'r') as f:
        content = json.load(f)
    return content['Measurements']

def prepare_data_tensor(point_cloud_np):
    pos = point_cloud_np[:, :3]
    pos = torch.tensor(pos, dtype=torch.float32)
    batch = torch.zeros(pos.shape[0], dtype=torch.long)  # 单样本
    return pos, batch

# def read_Label(path):
#     path = 'F:paper/read-papers/datasets/DRN-modify/OnlineChallenge/GroundTruth/GroundTruth_All_388_Images.json'
#     # path = '/home/ljs/workspace/eccv/FirstTrainingData/label/GroundTruth.json'
#     index = '50'
#     # contxt = json_load(path)
#     contxt = json.load(path)
#     print(contxt)
#     contxt = contxt['Measurements']
#     return contxt
#     # print(contxt)
#     # print(100*'*')
#     # item = 'Image%s' % index
#     # print(contxt[item])
#     # print(contxt[item]['Variety'])
#     # print(contxt[item]['RGBImage'])
#     # print(contxt[item]['DebthInformation'])
#     # print(contxt[item]['FreshWeightShoot'])
#     # print(contxt[item]['DryWeightShoot'])
#     # print(contxt[item]['Height'])
#     # print(contxt[item]['Diameter'])
#     # print(contxt[item]['LeafArea'])
#     # pass
#
# def _show_gd():
#     # path = '/home/ljs/workspace/eccv/FirstTrainingData/label/GroundTruth.json'
#     path = 'F:paper/read-papers/datasets/DRN-modify/OnlineChallenge/GroundTruth/GroundTruth_All_388_Images.json'
#     index = '50'
#     # contxt = json_load(path)
#     contxt = json.load(path)
#     print(contxt)
#     contxt = contxt['Measurements']
#     print(contxt)
#     print(100*'*')
#     item = 'Image%s' % index
#     print(contxt[item])
#     print(contxt[item]['Variety'])
#     print(contxt[item]['RGBImage'])
#     print(contxt[item]['DebthInformation'])
#     print(contxt[item]['FreshWeightShoot'])
#     print(contxt[item]['DryWeightShoot'])
#     print(contxt[item]['Height'])
#     print(contxt[item]['Diameter'])
#     print(contxt[item]['LeafArea'])
#     pass
#
#
# def show_gd():
#     # path = '/home/ljs/workspace/eccv/FirstTrainingData/label/GroundTruth.json'
#     path = 'F:paper/read-papers/datasets/DRN-modify/OnlineChallenge/GroundTruth/GroundTruth_All_388_Images.json'
#     index = '50'
#     per_label = list()
#     # contxt = json_load(path)
#     contxt = json.load(path)
#     print(contxt)
#     contxt = contxt['Measurements']
#     print(contxt)
#     print(100*'*')
#     item = 'Image%s' % index
#     print(contxt[item])
#     print(contxt[item]['Variety'])
#     print(contxt[item]['RGBImage'])
#     print(contxt[item]['DebthInformation'])
#     print(contxt[item]['FreshWeightShoot'])
#     print(contxt[item]['DryWeightShoot'])
#     print(contxt[item]['Height'])
#     print(contxt[item]['Diameter'])
#     print(contxt[item]['LeafArea'])
#     per_label.append(contxt[item]['FreshWeightShoot'])
#     per_label.append(contxt[item]['DryWeightShoot'])
#     per_label.append(contxt[item]['Height'])
#     per_label.append(contxt[item]['Diameter'])
#     per_label.append(contxt[item]['LeafArea'])
#     return per_label
    # pass





def read_pcd_pointclouds(file_path):
    # file_path = '/home/ljs/workspace/eccv/FirstTrainingData/out_4096/train/38.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points)
    # color_cloud = np.asarray(pcd.colors)*255
    color_cloud = np.asarray(pcd.colors)
    points = np.concatenate([point_cloud, color_cloud], axis=1)

    # 下采样或补全到1024点
    if points.shape[0] > 1024:
        indices = np.random.choice(points.shape[0], 1024, replace=False)
    else:
        indices = np.random.choice(points.shape[0], 1024, replace=True)


    # print(point_cloud.shape)
    # print(color_cloud.shape)
    # print(points.shape)
    # return points
    # print(color_cloud)
    # np.savetxt('38.txt', points, fmt='%10.8f') # Keep 8 decimal places
    # pass

    return points[indices]
# ========== 主测试流程 ==========
# def test_model():
def test_model_on_dataset(model, json_labels, ply_folder, image_ids, device):
    preds = []
    gts = []

    # ========== 加载标签 ==========
    # label_dict = read_label(label_json_path)
    for img_id in image_ids:
        item_key = f"Image{img_id}"
        if item_key not in json_labels:
            continue  # 跳过不存在的样本

        label_item = json_labels[item_key]
        gt_height = label_item["Height"]
        gts.append(gt_height)
    ply_path = os.path.join(ply_folder, f"{img_id}.ply")
    if not os.path.exists(ply_path):
        print(f"Missing file: {ply_path}")
        continue

    point_cloud_np = read_ply_pointcloud(ply_path)
    pos, batch = prepare_data_tensor(point_cloud_np)

    pos = pos.to(device)
    batch = batch.to(device)

    with torch.no_grad():
        output = model(pos, batch)
        pred_height = output.item()
        preds.append(pred_height)

    print(f"Image {img_id} - GT: {gt_height:.2f}, Pred: {pred_height:.2f}")


    return gts, preds


    # ========== 存储预测和标签 ==========
    # y_true = []
    # y_pred = []

    # for i in range(num_samples):
    #     image_id = f'Image{i}'
    #     pcd_path = os.path.join(pcd_dir, f'{i}.pcd')
    #
    #     if not os.path.exists(pcd_path):
    #         print(f"[跳过] 未找到文件: {pcd_path}")
    #         continue
    #
    #     if image_id not in label_dict:
    #         print(f"[跳过] 未在 JSON 中找到: {image_id}")
    #         continue

        # 读取点云数据
        # input_np = read_pcd_pointclouds(pcd_path)
        # input_tensor = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, N, 6)
        #
        # # 模型预测
        # with torch.no_grad():
        #     output = model(input_tensor)  # 假设返回 shape: (1, 1)
        #     prediction = output.item()
        #
        # # 提取 Ground Truth
        # true_height = label_dict[image_id]['Height']
        #
        # y_true.append(true_height)
        # y_pred.append(prediction)
        #
        # print(f"[样本 {i}] True: {true_height:.2f} | Pred: {prediction:.2f}")

    # ========== 可视化散点图 ==========

def plot_and_evaluate(gt_list, pred_list):
    mse = mean_squared_error(gt_list, pred_list)
    r2 = r2_score(gt_list, pred_list)

    print(f"\n✅ 测试完成：")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score (Accuracy): {r2:.4f}")

    plt.figure(figsize=(8, 6))
    # plt.scatter(y_true, y_pred, c='blue', alpha=0.6, label='Predicted vs True')
    plt.scatter(gt_list, pred_list, c='blue', alpha=0.7, label='Height')
    # plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Prediction')
    plt.plot([min(gt_list), max(pred_list)], [min(gt_list), max(gt_list)], 'r--', label='Ideal Prediction')
    plt.xlabel('Ground Truth Height')
    plt.ylabel('Predicted Height')
    plt.title('Plant Height Prediction (Regression)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # # ========== 打印误差 ==========
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    # mse = np.mean((y_true - y_pred) ** 2)
    # mae = np.mean(np.abs(y_true - y_pred))
    # print(f"\n✅ 测试完成\n均方误差 MSE: {mse:.4f}\n平均绝对误差 MAE: {mae:.4f}")
    #



# def mse_cal():
#     a = np.asarray([[1,2,4], [2,3,5]])
#     b = np.asarray([[0,0,4], [2,3,5]])
#     # print(a.shape)
#     print((a-b)*(a-b))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_csv', action='store_true', help='是否将结果保存为 CSV')
    args = parser.parse_args()

    #配置路径# ========== 参数配置 ==========
    model_path = 'F:/paper/read-papers/checkpoints/your_model.pt'  # 模型路径
    pcd_dir = 'F:/paper/read-papers/datasets/DRN-modify/test_pcd'  # PCD文件夹路径
    label_json_path = 'F:/paper/read-papers/datasets/DRN-modify/OnlineChallenge/GroundTruth/GroundTruth_All_388_Images.json'
    # num_samples = 388  # 样本总数
    image_ids = [str(i) for i in range(388)]  # 或自定义测试集
    # test_model()
    # mse_cal()

    # 主逻辑执行
    model = load_model(model_path, device)
    json_labels = load_labels(json_path)



    gts, preds = test_model_on_dataset(model, json_labels, ply_folder, image_ids, device)
    plot_and_evaluate(gts, preds)

    if args.save_csv:
        import pandas as pd

        df = pd.DataFrame({
            "ImageID": image_ids[:len(gts)],
            "GroundTruth": gts,
            "Prediction": preds,
            "AbsoluteError": np.abs(np.array(gts) - np.array(preds))
        })
        csv_path = "prediction_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ 预测结果已保存至: {csv_path}")
    # read_pcd_pointclouds()
    # show_gd()

    # file_path = '/home/ljs/workspace/eccv/FirstTrainingData/out_4096/train/38.pcd'
    # read_pcd_pointclouds(file_path)

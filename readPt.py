import torch
import os

# 加载.pt文件
DRN_path = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(DRN_path)
image_path = f'{root_directory}/OnlineChallenge'  # 将文件储存到OnlineChallenge目录中
savemodel_name = os.path.join(root_directory, 'weights', '1_epoch.pt')
content = torch.load(savemodel_name)

# 检查内容类型
if isinstance(content, dict):
    # 打印字典中的键
    print("Keys in the checkpoint:", content.keys())

    # 查看模型权重的键
    if 'model_state_dict' in content:
        model_state_dict = content['model_state_dict']
        print("Model state dict keys:", model_state_dict.keys())
else:
    # 如果内容是模型权重
    print("Model state dict keys:", content.keys())

import os.path as osp
import os
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
# from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
# from torch_geometric.data import DataLoader
from torch_geometric.nn import fps, radius, global_max_pool
from torch_geometric.nn.conv import PointConv
from DRN.show_data import MYData_Lettuce
from DRN.train_eval import run
from DRN.depth2Cloud import generate_ply_file
# from train_eval_origine import run


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=10)  # default: 32
parser.add_argument('--lr', type=float, default=0.001)  # defualt:0.001
parser.add_argument('--lr_decay_factor', type=float, default=0.5) # try 0.3
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio#用于最远点采样(FPS)的比例参数，决定了从点云中选择多少个点。
        self.r = r
        self.conv = PointConv(nn)#PointConv = PointNetConvPointConv 通过 edge_index 计算局部特征聚合，相当于对每个采样点 pos[idx] 在其邻域内进行特征提取。x = None 时，PointConv 通常会自动初始化点特征，并完成局部特征聚合

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)#radio=0.5:下采样的比例，即从原始点云中采样的点占原始点的比例。从原始点云中采样更具代表性的点。
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)#（领域搜索）radius:根据一个给定半径 r 在点云中搜索邻近点。它通常用于查找某个点的附近所有点，这在点云的特征提取和局部邻域计算中是非常常见的操作。在每个点的邻域内查找相邻点，形成邻接图。
        edge_index = torch.stack([col, row], dim=0)## 生成图结构的边，col：目标邻居点；row：原始目标点pos[idx]。torch.stack将输入的多个张量沿着新的维度进行堆叠。表示点与带点之间的连接关系（将邻接点和原始点堆叠，形成点云特征图的边索引。）
        x = self.conv(x, (pos, pos[idx]), edge_index)##卷积操作：根据点云的空间位置和邻接关系（通过 edge_index）进行局部特征聚合。
        pos, batch = pos[idx], batch[idx]#对应的点坐标 pos 和批次索引 batch 也更新为下采样后的结果。
        return x, pos, batch
#dim表示要检索大小的维度。功能：点云的下采样和局部特征提取。目的是在降维的同时提取更具判别力的局部特征。

class GlobalSAModule(torch.nn.Module):#全局特征提取
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)#全局最大池化是一种在卷积神经网络（CNN）中常用的池化技术，它的作用是将每个特征图（feature map）中的最大值提取出来，形成一个新的、更小的特征图。这种方法可以减少模型的参数数量，降低过拟合的风险，并保留重要的空间信息。
        pos = pos.new_zeros((x.size(0), 3))#为了将经过池化后的特征（通常丢失了原始位置）重新赋值一个位置张量。由于 global_max_pool 操作是全局性的，它将所有点的信息聚合为一个特征，而原始的点的位置已经不再有意义，因此我们用全零位置来代替。
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
#功能：用于提取点云的全局特征，结合全局最大池化操作，将点云从局部信息聚合为全局表征。

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, out_features):
        super(Net, self).__init__()
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        # self.lin2 = Lin(512, 256)
        self.lin2 = Lin(512, out_features)#out_features模型输出
        # self.lin3 = Lin(256, out_features)
        print('out_feature:',out_features)

    def forward(self, pos, batch):

        # sa0_out = (data.x, data.pos, data.batch)
        sa0_out = (None, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)#*sa0_out 将 sa0_out 中的内容解包并作为参数传递给 sa1_module，由sa模型处理后返回新的聚合特征，下采样坐标pos，以及新的对应批次索引batch。sa0_out = (x, pos, batch)。该段目的是在降维的同时提取更具判别力的局部特征。x（None→x） 变为经过特征提取后的点特征，pos 和 batch 表示下采样后的点云信息。
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        # return F.log_softmax(x, dim=-1)
        return x


class Net_LSTM(torch.nn.Module):
    def __init__(self, out_features):
        super(Net_LSTM, self).__init__()
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.input_size = 1  # feature points size or word size defualt:129
        self.out_size = 1  # the size of prediction for each word
        self.layer_num = 2
        # self.rnn_3 = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
        #                    bidirectional=True,
        #                    dropout=0.3,
        #                    # batch_first=True
        #                    )
        self.rnn = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
                                 bidirectional=True,
                                 dropout=0.3,  # defualt:0.3
                                 batch_first=True
                                 )#batch_first=True：意味着输入数据的形状是 (batch_size, seq_len, input_size)，让批量数据的维度在最前面，方便处理。

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)#AdaptiveAvgPool1d(1) 表示将输入的 序列长度（时间步）压缩到 1，即无论输入长度多少，输出始终是 (batch_size, hidden_size, 1)。这个操作相当于对所有时间步的特征求均值，得到一个全局特征。
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)#计算全局最大池化，取 LSTM 输出特征的 最大值，类似 CNN 中的 max pooling。AdaptiveMaxPool1d(1) 同样把时间步维度压缩成 1，输出形状 (batch_size, hidden_size, 1)。这个操作提取 最强的特征响应，更关注最大激活的特征值。

        self.lin1 = Lin(1024, 512)
        # self.lin2 = Lin(512, 256)
        self.lin2 = Lin(512, out_features)
        # self.lin3 = Lin(256, out_features)

    def forward(self, pos, batch):

        batch_size = batch.reshape(-1, 1024).shape[0]
        # 这里的1024是一个限定值（影响实验过程）
        #对输入的 batch 进行重塑（reshape）并计算新的形状的第一个维度的大小。这行代码的目的是计算批次大小 (batch size)。
        print(f'----batch_size = {batch_size}')
        # sa0_out = (data.x, data.pos, data.batch)
        sa0_out = (None, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out # x: (batch, 1024)

        x = x.reshape(batch_size, 1024, -1)#同上方1024一致
        print(f'--------------{x.shape}')
        #重新调整 x 的形状，使其成为 LSTM 输入的格式，其中：batch_size：批量大小 .1024：时间步（这里每个点相当于 LSTM 的时间步）.-1：自动计算通道数（特征维度）
        x, hc = self.rnn(x)
        x = self.avgpool(x)

        print('after_pooling:', x.shape)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = sel/home/ljs/PycharmProjects/dataf.lin3(x)
        # return F.log_softmax(x, dim=-1)#计算交叉熵损失
        # print(f'****,{out_features}')
        print('模型预测结果：', x)
        return x


if __name__ == '__main__':
    # 获取当前文件所在的目录
    DRN_path = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.dirname(DRN_path)
    image_path = f'{root_directory}/OnlineChallenge'#将文件储存到OnlineChallenge目录中
    # 该函数只在第一次调用时打开，用于生成ply文件，生成后可注释掉
    # generate_ply_file(DRN_path)
    # data_path = '/home/ljs/PycharmProjects/data'
    # data_name = ['FirstTrainingData']
    data_name = ['ori_data']
    # train_path = os.path.join(data_path,  train_data)
    # test_path = os.path.join(data_path, test_data)
    full_dataset = MYData_Lettuce(data_path=root_directory, data_name=data_name, points_num=1024)

    # 数据划分：70% train，15% val，15% test
    total_len = len(full_dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可复现
    )
    # train_dataset = MYData_Lettuce(data_path=root_directory, data_name=data_name, data_class='train', points_num=1024)

    # print('train_dataset num:', train_dataset.__len__())
    # test_dataset = MYData_Lettuce(data_path=root_directory, data_name=data_name, data_class='test', points_num=1024)

    # NUM_CLASS = 6
    out_features = 1
    # print(train_dataset.data[0])
    # model = Net(out_features)
    # model = Net(out_features)
    model = Net_LSTM(out_features)
    # print("x shape:", x.shape)  # 应该是 [num_nodes, num_features]
    # print("batch shape:", batch.shape)

    print('============================================')
    run(root_directory, train_dataset,val_dataset,test_dataset, model, args.epochs, args.batch_size, args.lr,
        args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay)




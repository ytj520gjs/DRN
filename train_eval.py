import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import csv
import os
# from apex import amp
from torch.cuda import amp
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable


def csv_writer(file_path, item='Predict_Number'):
    f = open(file_path, 'w', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['FileName', item])#向 CSV 文件中写入一行数据，包含两个字段：'FileName' 和 item。
    return writer, f


def get_batch(batch_size, points_num=1024):
    arr = torch.ones(points_num, dtype=int)#创建points_num为全1整数张量
    arr_sum = torch.zeros(points_num, dtype=int)
    for i in range(1, batch_size):
        arr_sum = torch.cat((arr_sum, arr*i))#cat:拼接将 arr_sum 和 arr*i 这两个张量沿着第一个维度（即行的方向）拼接起来，扩展 arr_sum 的大小
    return arr_sum
#arr_sum 包含了所有拼接后的张量，长度为 batch_size * points_num，如果batch-size=2时，arr_sum=2048个值
# .module.state_dict()
def save_net(fname, net):#文件名，及神经网络。保存神经网络模型参数
    torch.save(net.state_dict(), fname)#net.state_dict()：提取模型的参数（如权重和偏置），并以字典形式保存。
    net.train()
    if torch.cuda.is_available():
        net.cuda()

#为什么使用 .state_dict() 而不是 torch.save(model, ...)？
# ✅ 轻量化：仅保存模型参数，避免冗余信息。
# ✅ 灵活性：可以将参数迁移到不同的模型结构上（例如微调时）。
# ✅ 兼容性：版本升级或模型调整时，仍可轻松加载。

    # .module.state_dict()
def run(root_directory, train_dataset, test_dataset, model, epochs, batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay):#负责损失函数定义，模型的初始化与转移到指定设备，优化器的设置，结果文件的创建
    criterion = nn.MSELoss()#标准
    model = model.to(device)#模型的设备转移
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #初始化Adam优化器。Adam ➔ 自适应矩估计优化器，model.parameters() ➔ 获取模型中所有可训练参数。
    # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)#初始化Adam优化器。
    # Adam ➔ 自适应矩估计优化器，model.parameters() ➔ 获取模型中所有可训练参数。lr ➔ 学习率（控制参数更新的步幅，较小的值收敛更稳定但较慢，
    # 较大值收敛更快但可能不稳定）。weight_decay ➔ 权重衰减（L2 正则化），有助于防止过拟合。
    csv_filePath = os.path.join(root_directory, 'result',
                                '%s_test_accuracy_' % (2) + time.strftime("%d-%m-%Y-%H-%M-%S") + '.csv')
    #time.strftime：动态生成时间戳，确保文件名唯一，时间戳的生成，日期%d，月份%m，年份%Y，%H小时创建一个 CSV 文件以记录模型训练的性能指标（如准确率、损失）
    writer_ljs, f = csv_writer(csv_filePath)
    # writer, f = csv_writer(csv_filePath)
    # apex = True
    #
    # cuda = True
    # scaler = amp.GradScaler(enabled=cuda)

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)#记录了一些数据形式的要求
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # print('test_loader:', test_loader)

    for epoch in range(1, epochs + 1):#确保迭代训练中，每个epoch可以循环到
        if torch.cuda.is_available():
            torch.cuda.synchronize()#确保CUDA设备（GPU）的可用性
        t_start = time.perf_counter()#性能计数器，通常用于计算经过的时间。
        # with amp.autocast(enabled=cuda):
        train_loss = train(model, optimizer, train_loader, device, criterion)#输出平均损失
        test_acc, csv_out, csv_label_out = test(model, test_loader, device, epoch)
        savemodel_name = os.path.join(root_directory, 'weights', '%s_epoch.pt'%epoch)
        save_net(savemodel_name, model)
        csv_out = np.concatenate((csv_out, [test_acc.detach().cpu().numpy()], [train_loss.detach().cpu().numpy()]))
        # csv_out = np.asarray(list(csv_out).append(test_acc))
        # print('csv_out:',  csv_out)
        if epoch == 1:
            writer_ljs.writerow(csv_label_out)
        writer_ljs.writerow(csv_out)
        if torch.cuda.is_available():
            torch.cuda.synchronize()#如果GPU可用，这行代码会同步CUDA操作，即等待所有GPU上的操作完成。

        t_end = time.perf_counter()#测试时间差t_start 是代码开始执行时的时间戳。t_end 是代码结束执行时的时间戳。

        print('Epoch: {:03d}, Test: {:.4f}, Duration: {:.2f}'.format(
            epoch, test_acc, t_end - t_start))

        if epoch % lr_decay_step_size == 0:#如果余数为 0，说明已经达到了指定的周期，应该进行学习率衰减。
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

#这段代码的作用是实现学习率（learning rate）的衰减。在训练过程中，当经过一定数量的训练周期（epoch）后，学习率会按一定比例减小，以帮助模型在训练后期更稳定地收敛。

    f.close()


def train(model, optimizer, train_loader, device, criterion):
    # criterion = nn.MSELoss()
    # np.random.seed(0)
    model.train()#将模型设置为训练模式
    total_loss = 0#初始化总损失
    for data, label in train_loader:
        optimizer.zero_grad()#清除先前梯度，以防止在多次训练时梯度的累积。每次计算梯度时，都会用之前的梯度更新参数，因此每次训练开始前都要清除。
        batch_size = data.shape[0]# 获取当前批次的大小（数据的第一维度是批次大小）
        # data = data.to(device)
        # 打乱点云顺序
        for sample in data:
            np.random.shuffle(sample)# 对每个点云样本进行打乱，打乱的是样本的顺序
        if np.random.randint(0, 2):
            data[:, :, 0] = -data[:, :, 0]#翻转X轴（水平翻转）
            if np.random.randint(0, 2):
               data[:, :, 1] = -data[:, :, 1]#如果上一步生成的随机数是 1，则对数据的第二个维度（如 y 坐标）进行翻转，执行垂直翻转。翻转y轴（垂直翻转）
       #通过这种随机翻转操作，可以增加数据的多样性，作为一种数据增强方法，提高模型的泛化能力。
       
       
        #if np.random.randint(0, 2):
        #     data[:, :, 0] = data[:, :, 0] + np.random.randint(-3, 3) * 0.01
        #     data[:, :, 1] = data[:, :, 1] + np.random.randint(-3, 3) * 0.01
        #     data[:, :, 2] = data[:, :, 2] + np.random.randint(-3, 3) * 0.01
        # print('train_data.shape', data.shape)
        data = Variable(data.float().reshape(-1, 3).to(device))
        #data 被包装在 Variable 中，这样它就可以参与计算图并支持自动求导。
        # data = Variable(data.float().reshape(-1, 3).to(device))#data 被包装在 Variable 中，
        # 这样它就可以参与计算图并支持自动求导。这行代码常见于处理输入数据（如点云数据、图像数据等）时，需要进行类型转换、形状调整和设备迁移。


        # label = label.long().to(device)
        label = label.float().to(device)
        # label = label[:, 2:4]
        # print('label:', label[:, 2:4])
        batch = get_batch(batch_size).to(device)
        # label = Variable(label[:, 3])
        label = Variable(torch.unsqueeze(label[:, 2], dim=1))#label[:, 2]从 label 张量中提取所有样本的第 3 列数据。而后torch.unsqueeze就是使其在原来维度（行）的基础上加入一个新的维度，变成列向量
        # print('trian_label.shape:', label.shape)
        # out = copy.copy(model(data, batch))
        # with amp.autocast(enabled=True):
        out = model(data, batch)#模型输出
        loss = criterion(out, label)#模型输出和实际标签之间的损失，返回一个标量值，表示预测值和真实值之间的差异。这个值会被用来指导模型的优化过程，通过反向传播来更新模型的参数。
        # loss = F.nll_loss(out, data.y)
        total_loss += loss
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        loss.backward()#反向传播：深度学习中一种常用的优化方法，用来计算损失函数相对于模型参数的梯度。
        optimizer.step()#更新模型参数
        # with amp.autocast(enabled=True):
    print('train_loss:', total_loss/len(train_loader))#输出每个训练周期结束时的平均训练损失。
    return total_loss/len(train_loader)


def test(model, test_loader, device, epoch):
    model.eval()#设置模型评估模式
    # print('epoch:', epoch)
    # correct = 0
    total_mse = 0
    csv_out_list = torch.tensor(list()).to(device)
    csv_out_label = torch.tensor(list()).to(device)
    for data, label in test_loader:
        # data = data.to(device)
        batch_size = data.shape[0]
        # print(' batch_size :',  batch_size )
        # 打乱点云顺序
        # for sample in data:
        # np.random.shuffle(sample)
        data = data.float().reshape(-1, 3).to(device)
        label = label.float().to(device)
        batch = get_batch(batch_size).to(device)
        # label = label[:, 3]
        # label = Variable(label[:, 3])
        label = Variable(torch.unsqueeze(label[:, 2], dim=1))
        out = copy.copy(model(data, batch))#通过使用 copy.copy() 创建一个新的对象副本，可以保证原始输出不被改变。
        if epoch == 1:
           csv_label = torch.squeeze(label, dim=-1)
           csv_out_label = torch.cat((csv_out_label,csv_label))#csv_out_label 逐渐包含所有 epoch 中的标签数据。

        csv_out = torch.squeeze(out, dim=-1)#移除batch的值
        csv_out_list = torch.cat((csv_out_list, csv_out))#？没懂，为什么是对所有batch的拼接，输出的是batch带来的信息，所以拼接指是信息数据的拼接而不是batch
        # print(csv_out)
        # csv_out
        # print(csv_out_list.cpu().numpy())
        loss = torch.nn.functional.mse_loss(out, label)
        # pred = model(data.pos, data.batch).max(1)[1]
        total_mse += loss
    test_acc = total_mse / len(test_loader)
    # model.train()
    return test_acc, csv_out_list.detach().cpu().numpy(), csv_out_label.detach().cpu().numpy()
#csv_out_list 是模型的输出，最初是一个 torch.Tensor。
# .detach()：从计算图中分离张量，防止反向传播中梯度的传播（用于测试时，不再需要梯度）。防止梯度传播，节省内存，避免不必要的计算图追踪。
# .cpu()：将张量从 GPU 移动到 CPU（NumPy 不支持直接从 GPU 转换数据）。NumPy 无法直接处理 GPU 数据。
# .numpy()：将 PyTorch 张量转换为 NumPy 数组，方便进一步分析、保存或可视化
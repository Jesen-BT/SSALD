from skmultiflow.data.file_stream import FileStream
import torch
import matplotlib.pyplot as plt
from SSALD import SSALD, def_loss_fn

name = ['SINE']
for _ in name:
    file_name =  _ +".csv"
    stream = FileStream(file_name)
    data, label = stream.next_sample(200)
    rbf = SSALD(20, 1, window_size=200, label_budget=0.05)

    data = torch.tensor(data, dtype=torch.float32)
    label = torch.unsqueeze(torch.tensor(label, dtype=torch.float32), 1)
    rbf.model_initialization(data, label)
    loss_fn = def_loss_fn()
    params = rbf.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-3)

    data_window, label_window, weight = rbf.return_window()

    for i in range(100):
        optimizer.zero_grad()
        y = rbf.forward(data_window)
        loss = loss_fn.forward(y, label_window, weight)
        loss.backward()
        optimizer.step()

    count_rbf = 0
    data_size = 0

    for i in range(y.size(0)):
        if y[i, 0] > 0.5:
            y[i, 0] = 1.
        else:
            y[i, 0] = 0.

    result_rbf = []
    t_list = []
    index = -1

    while stream.has_more_samples():
        params = rbf.parameters()
        optimizer = torch.optim.Adam(params, lr=1e-3)
        data_size = data_size + 1
        new_x, new_y = stream.next_sample()

        new_x = torch.tensor(new_x, dtype=torch.float32)
        new_y = torch.tensor(new_y)
        new_y = torch.unsqueeze(new_y, 1)

        y = rbf.forward(new_x)

        if y[0, 0] > 0.5:
            y[0, 0] = 1.
        else:
            y[0, 0] = 0.

        if y[0, 0] == new_y[0, 0]:
            count_rbf = count_rbf + 1

        rbf.partial_fit(new_x, new_y)
        data_window, label_window, weight = rbf.return_window()

        loss_dif = 1
        round = 0

        while loss_dif > 0.0001 and round < 100:
            optimizer.zero_grad()
            y = rbf.forward(data_window)
            pre_loss = loss_fn.forward(y, label_window, weight)
            pre_loss.backward()
            optimizer.step()

            y = rbf.forward(data_window)
            aft_loss = loss_fn.forward(y, label_window, weight)

            loss_dif = pre_loss - aft_loss
            round = round + 1


        if data_size % 100 == 0.:
            result_rbf.append(count_rbf / data_size)
            t_list.append(data_size)
            plt.plot(t_list, result_rbf, c='r', ls='-', marker='o', mec='b', mfc='w', label='SSALD')
            if index == -1:
                plt.legend()
                index = 0
            plt.pause(0.1)

    plt.show()


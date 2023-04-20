import torch, random
import torch.nn as nn
import numpy as np
import warnings

def Euclidean_Distances(a,b):
    dis = torch.sqrt(torch.sum((a-b)**2, dim=1))
    return dis

def kernel_fun(a, b, sigma):
     A = torch.sum((a-b)**2, dim=1)
     B = A/(2*sigma**2)
     C = torch.exp(-B)
     return C

torch.manual_seed(42)

class SSALD(nn.Module):

    def __init__(self, num_center, n_out, window_size, label_budget=0.1):

        super(SSALD, self).__init__()
        self.n_out = n_out
        self.num_centers = num_center

        self.window_size = window_size
        self.label_budget = label_budget
        self.data_window = torch.zeros(window_size, 1)
        self.label_window = torch.zeros(window_size, 1)
        self.label_index = torch.zeros(window_size, 1)
        self.plabel_window = torch.zeros(window_size, 1)

        self.labeled_data = torch.zeros(int(window_size/2), 1)
        self.labeled_data_y = torch.zeros(int(window_size/2), 1)

        self.labeled_weight = torch.zeros(int(window_size/2), 1)

        self.sigma = 0

    def kernel_fun(self, batches):
        n_input = batches.size(0)
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def window_update(self, data, label):
        self.data_window = torch.cat([self.data_window[data.size(0):, :], data], dim=0)
        self.label_window = torch.cat([self.label_window[data.size(0):, :], label], dim=0)

        new_index = torch.zeros(data.size(0), 1)
        self.label_index = torch.cat([self.label_index[data.size(0):, :], new_index], dim=0)

    def return_window(self):
        self.pseudo_label()
        zero_indices = torch.where(self.label_index == 0)[0]
        unlabeled = self.data_window[zero_indices, :]
        Plabel = self.plabel_window[zero_indices, :]

        unlabel_weight = torch.ones(len(unlabeled), 1)

        data = torch.cat([unlabeled, self.labeled_data], dim=0)
        label = torch.cat([Plabel, self.labeled_data_y], dim=0)
        weight = torch.cat([unlabel_weight, self.labeled_weight], dim=0)

        return data, label, weight

        # return torch.cat([self.data_window, self.labeled_data], dim=0), torch.cat([self.plabel_window, self.labeled_data_y], dim=0)

    def update_sigma(self):
        distance = 0
        for i in range(self.num_centers):
            dis = Euclidean_Distances(self.centers[i], self.centers)
            dis = torch.max(dis)
            if dis > distance:
                distance = dis
        self.sigma = distance/torch.sqrt(torch.tensor(2*self.num_centers))
        self.beta = torch.ones(1, self.num_centers) * self.sigma

    def model_initialization(self, data, label):
        self.data_window = torch.zeros([self.window_size, data.size(1)], dtype=torch.float32)
        self.label_window = torch.zeros([self.window_size, self.n_out], dtype=torch.float32)
        self.label_index = torch.zeros([self.window_size, 1], dtype=torch.float32)

        self.window_update(data, label)
        self.labeled_data = self.data_window[int(self.window_size/2):, :]
        self.labeled_data_y = self.label_window[int(self.window_size/2):, :]

        self.labeled_weight = torch.ones(len(self.labeled_data), 1)
        for i in range(len(self.labeled_data)):
            for i in range(len(self.labeled_data) - 2, -1, -1):
                self.labeled_weight[i] = self.labeled_weight[i + 1] * (1 - 1 / len(self.labeled_data))

        self.label_index = torch.ones((self.window_size,1), dtype=torch.float32)
        index = torch.LongTensor(random.sample(range(self.data_window.size(0)), self.num_centers)) #随机选择center
        self.centers = torch.index_select(self.data_window, 0, index)
        self.dim_centure = self.centers.size(1)

        self.update_sigma()
        self.linear = nn.Sequential(nn.Linear(self.num_centers + self.dim_centure, self.n_out, bias=True)
                                    , nn.Sigmoid())
        self.initialize_weights()

    def sample_selection(self):  # 根据平滑度损失作为概率，来随机选择，在保证多样性的时候最大化平滑度。
        num = int(self.window_size * self.label_budget)
        label_probability = 1. - self.label_index.sum() / num

        if random.random() < label_probability:
            pre = self.forward(self.data_window)
            index = torch.arange(self.window_size, requires_grad=False).detach().numpy()
            self.pseudo_label()
            smooth_loss = -(self.plabel_window * torch.log(pre + 1e-8) + (
                    1 - self.plabel_window) * torch.log(1 - pre + 1e-8))
            labeled_index = torch.squeeze(self.label_index == 1., 1)
            smooth_loss[labeled_index] = 0
            smooth_loss = torch.squeeze(smooth_loss)
            smooth_loss = smooth_loss / torch.sum(smooth_loss)
            smooth_loss = smooth_loss.detach().numpy()

            rand_index = np.random.choice(index, p=smooth_loss)

            self.label_index[rand_index] = 1
            self.labeled_data = torch.cat(
                [self.labeled_data[1:, :], torch.unsqueeze(self.data_window[rand_index], dim=0)],
                dim=0)
            self.labeled_data_y = torch.cat(
                [self.labeled_data_y[1:, :], torch.unsqueeze(self.label_window[rand_index], dim=0)], dim=0)

    def pseudo_label(self):
        self.plabel_window = torch.zeros((self.window_size, 1), requires_grad=False)

        for i in range(self.window_size):
            if self.label_index[i, 0] == 1:
                self.plabel_window[i,0] = self.label_window[i,0]
            else:
                s = kernel_fun(self.labeled_data, self.data_window[i], sigma=self.sigma)
                s = torch.unsqueeze(s, dim=1)
                slable = self.labeled_weight*s*self.labeled_data_y
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.plabel_window[i, 0] = torch.tensor(torch.sum(slable) / torch.sum(s*self.labeled_weight))

    def calculate_representativeness(self):
        representative = torch.zeros((self.num_centers, 1))
        for i in range(self.num_centers):
            rep_ci = kernel_fun(self.centers[i], self.centers, sigma=self.sigma)
            rep_ci[i] = 0.
            representative[i, 0] = 1 - torch.max(rep_ci)

        for i in range(self.num_centers):
            eff_ci = kernel_fun(self.centers[i], self.data_window, sigma=self.sigma)
            eff_ci = torch.mean(eff_ci)
            representative[i, 0] = eff_ci + representative[i, 0]
        return representative

    def center_adjustment(self):
        representative = self.calculate_representativeness()

        min_index = representative.argmin(dim=0)
        round = 0
        while representative[min_index] < 0.5 and round < self.num_centers:
            representative = torch.zeros((self.window_size, 1))
            for i in range(self.window_size):
                rep_ci = kernel_fun(self.data_window[i], self.centers, sigma=self.sigma)
                rep_ci[min_index] = 0.
                representative[i, 0] = 1 - torch.max(rep_ci)

            for i in range(self.window_size):
                eff_ci = kernel_fun(self.data_window[i], self.data_window, sigma=self.sigma)
                eff_ci = torch.mean(eff_ci)
                representative[i, 0] = eff_ci + representative[i, 0]

            max_index = representative.argmax(dim=0)
            self.centers[min_index] = self.data_window[max_index]
            representative = self.calculate_representativeness()
            min_index = representative.argmin(dim=0)

            round = round + 1

        self.update_sigma()

    def partial_fit(self, data, label):
        self.window_update(data, label)
        self.sample_selection()
        self.center_adjustment()

class def_loss_fn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, weight):
        corss_loss = -torch.mean(y_true * torch.log(y_pred + 1e-8) * weight + (1 - y_true) * torch.log(1 - y_pred + 1e-8) * weight)
        loss = corss_loss
        return loss
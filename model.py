import torch.nn as nn
import itertools
import evaluation
from sklearn.cluster import KMeans
from util import *

class Generator(nn.Module):
    def __init__(self, output_dim, nb_units=[256, 256, 256], batchnorm=False):
        super(Generator, self).__init__()
        self.output_dim = output_dim
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = nn.Sequential()

        """ Builds the FC stacks. """
        for i in range(len(nb_units)):
            if i == (len(nb_units) - 1):
                units = self.output_dim
                fc_layer = nn.Linear(nb_units[i], units)
                self.all_layers.add_module(f"fc_layer_{i}", fc_layer)
            else:
                units = self.nb_units[i + 1]
                fc_layer = nn.Linear(nb_units[i], units)
                norm_layer = nn.BatchNorm1d(units) if batchnorm else nn.Identity()
                self.all_layers.add_module(f"fc_layer_{i}", fc_layer)
                self.all_layers.add_module(f"norm_layer_{i}", norm_layer)
                self.all_layers.add_module(f"leakyrelu_{i}", nn.LeakyReLU(0.2))

    def forward(self, inputs, training=True):
        output = self.all_layers(inputs)
        return output

class Discriminator(nn.Module):
    def __init__(self, nb_units=[256, 256], batchnorm=True):
        super(Discriminator, self).__init__()
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = nn.Sequential()

        """Builds the FC stacks."""
        for i in range(len(nb_units)):
            if i == (len(nb_units) - 1):
                units = 1
                fc_layer = (nn.Linear(nb_units[i], units))
                self.all_layers.add_module(f"fc_layer_{i}", fc_layer)
            else:
                units = self.nb_units[i + 1]
                fc_layer = (nn.Linear(nb_units[i], units))
                norm_layer = nn.BatchNorm1d(units) if batchnorm else nn.Identity()
                self.all_layers.add_module(f"fc_layer_{i}", fc_layer)
                self.all_layers.add_module(f"norm_layer_{i}", norm_layer)
                self.all_layers.add_module(f"tanh_layer_{i}", nn.Tanh())

    def forward(self, inputs):
        output = self.all_layers(inputs)
        return output

class Model():
    def __init__(self, config):
        self._config = config
        self._latent_dim = config['latent']
        self.z_sampler = Gaussian_sampler(mean=np.zeros(self._latent_dim), sd=1.0)
        self.fuse = Fuse()

        self.gx_1 = Generator(self._latent_dim, config['generators']['arch2'], config['generators']['batchnorm'])
        self.gx_2 = Generator(self._latent_dim, config['generators']['arch1'], config['generators']['batchnorm'])
        self.dx_1 = Discriminator(config['discriminators']['arch1'], config['generators']['batchnorm'])
        self.dx_2 = Discriminator(config['discriminators']['arch2'], config['generators']['batchnorm'])

        self.gx = Generator(self._config['training']['dim_all'], config['generators']['arch3'], config['generators']['batchnorm'])
        self.dx = Discriminator(config['discriminators']['arch3'], config['generators']['batchnorm'])
        self.dz = Discriminator(config['discriminators']['arch4'], config['generators']['batchnorm'])

        self.Contrastive_loss = InstanceLoss(temperature=0.2)

    def to_device(self, device):
        self.gx_1.to(device)
        self.gx_2.to(device)
        self.gx.to(device)
        self.dx_1.to(device)
        self.dx_2.to(device)
        self.dx.to(device)
        self.dz.to(device)
        self.fuse.to(device)

    def train(self, alp, config, X1, X2, Y_list, mask, device):
        Y_list = torch.tensor(Y_list).int().to(device).squeeze(dim=0).unsqueeze(dim=1)
        acc, nmi, ari = self.train_latent(alp, X1, X2, Y_list, mask,  device)

        return acc, nmi, ari

    def evaluation(self, alp, X1, X2, Y_list, mask, device):
        with torch.no_grad():
            self.gx_1.eval(), self.gx_2.eval(), self.gx.eval()
            fill_num = 5
            "Missing Part Inference"
            z1 = self.gx_1(X1)
            z2 = self.gx_2(X2)
            recover_out0 = (torch.empty_like(z1)).to(device)
            recover_out1 = (torch.empty_like(z2)).to(device)
            Z = self.z_sampler.get_batch(len(X1))
            Z = torch.from_numpy(Z).to(device)
            rec_x = self.gx(Z)
            _, n = X1.size()
            _, m = X2.size()
            rec_x1 = self.gx_1(rec_x[:, 0:n])
            rec_x2 = self.gx_2(rec_x[:, n:(n+m)])
            C = euclidean_dist(z1, z2)
            row_idx = C.argsort()
            col_idx = (C.t()).argsort()
            M = torch.logical_and((mask[:, 0].repeat(len(X1), 1)).t(), mask[:, 1].repeat(len(X1), 1))
            for i in range(len(X1)):
                idx0 = col_idx[i, :][M[col_idx[i, :], i]]
                idx1 = row_idx[i, :][M[i, row_idx[i, :]]]
                if len(idx1) != 0 and len(idx0) == 0:
                    avg_fill_g = rec_x2[idx1[0:fill_num], :].sum(dim=0)
                    avg_fill = (z1[idx1[0:fill_num], :].sum(dim=0) + alp * avg_fill_g) / (fill_num * 2)
                    recover_out0[i, :] = z1[i, :]
                    recover_out1[i, :] = avg_fill
                elif len(idx0) != 0 and len(idx1) == 0:
                    avg_fill_g = rec_x1[idx0[0:fill_num], :].sum(dim=0)
                    avg_fill = (z2[idx0[0:fill_num], :].sum(dim=0) + alp * avg_fill_g) / (fill_num * 2)
                    recover_out0[i, :] = avg_fill
                    recover_out1[i, :] = z2[i, :]
                elif len(idx0) != 0 and len(idx1) != 0:
                    recover_out0[i, :] = z1[i, :]
                    recover_out1[i, :] = z2[i, :]
            latent_fusion = self.fuse(recover_out0, recover_out1, recover_out0, recover_out1)
            latent_fusion = latent_fusion.cpu().detach().numpy()
            score = evaluation.clustering([latent_fusion], Y_list[:, 0, 0].cpu().detach().numpy())

        return score

    def train_dis(self, data_x1, data_x2, data_z, Z):
        data_x = torch.cat([data_x1, data_x2], dim=1)

        data_z1 = self.gx_1(data_x1)
        data_z2 = self.gx_2(data_x2)

        data_dz11 = self.dx_1(data_z1)
        data_dz12 = self.dx_1(data_z2)
        data_dz22 = self.dx_2(data_z2)
        data_dz21 = self.dx_2(data_z1)

        loss_dis1 = (torch.mean((0.9 * torch.ones_like(data_dz11) - data_dz11) ** 2) + torch.mean((0.1 * torch.ones_like(data_dz12) - data_dz12) ** 2)) / 2.0
        loss_dis2 = (torch.mean((0.9 * torch.ones_like(data_dz22) - data_dz22) ** 2) + torch.mean((0.1 * torch.ones_like(data_dz21) - data_dz21) ** 2)) / 2.0

        data_x_ = self.gx(Z)
        data_dx = self.dx(data_x)
        data_dx_ = self.dx(data_x_)

        data_dz = self.dz(data_z)
        data_dz_ = self.dz(Z)

        dis_loss_x = (torch.mean((0.9 * torch.ones_like(data_dx) - data_dx) ** 2) \
                      + torch.mean((0.1 * torch.ones_like(data_dx_) - data_dx_) ** 2)) / 2.0
        dis_loss_z = (torch.mean((0.9 * torch.ones_like(data_dz) - data_dz) ** 2) \
                      + torch.mean((0.1 * torch.ones_like(data_dz_) - data_dz_) ** 2)) / 2.0
        dis_loss = dis_loss_x + dis_loss_z

        loss_dis = loss_dis1 + loss_dis2 + dis_loss

        return loss_dis

    def train_gen(self, data_x1, data_x2, Z):
        data_x = torch.cat([data_x1, data_x2], dim=1)
        data_x_ = self.gx(Z)
        data_dx_ = self.dx(data_x_)

        l2_loss_x = torch.mean((data_x - data_x_) ** 2)
        loss_gx = torch.mean((0.9 * torch.ones_like(data_dx_) - data_dx_) ** 2)

        g_e_loss = loss_gx + l2_loss_x

        return g_e_loss

    def train_latent(self, alp, x1, x2, Y_list, mask, device):
        optimizer_d = torch.optim.Adam(
            itertools.chain(self.dx_1.parameters(), self.dx_2.parameters(), self.dx.parameters(), self.dz.parameters()), lr=self._config['training']['lr'])
        optimizer_g = torch.optim.Adam(
            itertools.chain(self.gx.parameters()), lr=self._config['training']['lr'])
        optimizer_c = torch.optim.Adam(
            itertools.chain(self.gx_1.parameters(), self.gx_2.parameters()), lr=0.0005)

        mask = torch.from_numpy(mask).to(device)
        flag_1 = (torch.LongTensor([1, 1]).to(device) == mask).int()
        Y_list = torch.tensor(Y_list).int().to(device).squeeze(dim=0).unsqueeze(dim=1)
        X1, X2, X3, X4 = x1, x2, flag_1[:, 0], flag_1[:, 1]
        best_acc, best_ari, best_nmi = 0, 0, 0

        with torch.autograd.set_detect_anomaly(True):
            for epochs in range(self._config['training']['epoch']):
                dg_freq = self._config['g_d_freq']
                loss_all = 0
                fill_num = 5
                for batch_x1, batch_x2, x1_index, x2_index, ma, batch_No in next_batch(X1, X2, X3, X4, mask, self._config['training']['batch_size']):
                    z1 = self.gx_1(batch_x1)
                    z2 = self.gx_2(batch_x2)
                    recover_out0 = (torch.empty_like(z1)).to(device)
                    recover_out1 = (torch.empty_like(z2)).to(device)
                    ZZ = self.z_sampler.get_batch(len(batch_x1))
                    ZZ = torch.from_numpy(ZZ).to(device)
                    with torch.no_grad():
                         rec_x = self.gx(ZZ)
                    _, n = batch_x1.size()
                    _, m = batch_x2.size()
                    rec_x1 = self.gx_1(rec_x[:, 0:n])
                    rec_x2 = self.gx_2(rec_x[:, n:(n+m)])
                    C = euclidean_dist(z1, z2)
                    row_idx = C.argsort()
                    col_idx = (C.t()).argsort()
                    M = torch.logical_and((ma[:, 0].repeat(len(batch_x1), 1)).t(),  ma[:, 1].repeat(len(batch_x1), 1))
                    for i in range(len(batch_x1)):
                        idx0 = col_idx[i, :][M[col_idx[i, :], i]]
                        idx1 = row_idx[i, :][M[i, row_idx[i, :]]]
                        if len(idx1) != 0 and len(idx0) == 0:
                            avg_fill_g = rec_x2[idx1[0:fill_num], :].sum(dim=0)
                            avg_fill = (z1[idx1[0:fill_num], :].sum(dim=0) + alp * avg_fill_g) / (fill_num * 2)
                            recover_out0[i, :] = z1[i, :]
                            recover_out1[i, :] = avg_fill
                        elif len(idx0) != 0 and len(idx1) == 0:
                            avg_fill_g = rec_x1[idx0[0:fill_num], :].sum(dim=0)
                            avg_fill = (z2[idx0[0:fill_num], :].sum(dim=0) + alp * avg_fill_g) / (fill_num * 2)
                            recover_out0[i, :] = avg_fill
                            recover_out1[i, :] = z2[i, :]
                        elif len(idx0) != 0 and len(idx1) != 0:
                            recover_out0[i, :] = z1[i, :]
                            recover_out1[i, :] = z2[i, :]

                    z1.copy_(recover_out0)
                    z2.copy_(recover_out1)

                    if dg_freq == 0:
                        dg_freq = self._config['g_d_freq']
                        if (batch_No == (self._config['g_d_freq'] + 1)) & (epochs == 0):
                            data_z = self.fuse(z1, z2, z1, z2)
                            estimator = KMeans(n_clusters=self._config['training']['cluster'], init='k-means++')
                            estimator.fit(data_z.cpu().detach().numpy())
                            centers = estimator.cluster_centers_
                            fea_cluster = torch.from_numpy(centers).to(device)
                            prev_centers = fea_cluster.T
                        else:
                            data_z = self.fuse(torch.mm(z1, prev_centers), torch.mm(z2, prev_centers), z1, z2)
                            estimator = KMeans(n_clusters=self._config['training']['cluster'], init=prev_centers.T.cpu().detach().numpy(), n_init=1)
                            data_zz = data_z.cpu().detach().numpy()
                            estimator.fit(data_zz)
                            centers = estimator.cluster_centers_
                            fea_cluster = torch.from_numpy(centers).to(device)
                            prev_centers = fea_cluster.T
                        Z = self.z_sampler.get_batch(len(batch_x1))
                        Z = torch.from_numpy(Z).to(device)
                        dis_loss = self.train_dis(batch_x1, batch_x2, data_z, Z)
                        optimizer_d.zero_grad()
                        dis_loss.backward(retain_graph=True)
                        optimizer_d.step()
                        loss_all = loss_all + dis_loss.item()
                    else:
                        dg_freq -= 1
                        Z = self.z_sampler.get_batch(len(batch_x1))
                        Z = torch.from_numpy(Z).to(device)
                        gen_loss = self.train_gen(batch_x1, batch_x2, Z)
                        optimizer_g.zero_grad()
                        gen_loss.backward(retain_graph=True)
                        optimizer_g.step()
                        loss_all = loss_all + gen_loss.item()

                    data_dx1 = self.dx_1(z2)
                    data_dx2 = self.dx_2(z1)
                    ins_loss = self.Contrastive_loss(z1, z2, data_dx1, data_dx2)
                    optimizer_c.zero_grad()
                    ins_loss.backward(retain_graph=True)
                    optimizer_c.step()
                    loss_all = loss_all + ins_loss.item()

                if (epochs % 1 == 0):
                    score = self.evaluation(alp, x1, x2, Y_list, mask, device)
                    acc = score[0]['kmeans']['ACC']
                    nmi = score[0]['kmeans']['NMI']
                    ari = score[0]['kmeans']['ARI']

                    output = "Epoch latent: {:.0f}/{} " \
                             "==> Loss = {} " \
                        .format(epochs, self._config['training']['epoch'], loss_all)
                    print(output)

                    if (acc > best_acc):
                        best_acc = acc
                        best_ari = ari
                        best_nmi = nmi

        return [best_acc, best_nmi, best_ari]


class Fuse(torch.nn.Module):
    def __init__(self):
        super(Fuse, self).__init__()

    def forward(self, feature1, feature2, z1, z2):
        w1 = torch.var(feature1)
        w2 = torch.var(feature2)
        a1 = w1 / (w1 + w2)
        a2 = 1 - a1
        fused_feature = torch.add(z1 * a1, z2 * a2)

        return fused_feature

class InstanceLoss(nn.Module):
    def __init__(self, batch_size=256, temperature=1):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j, data_dx1, data_dx2):
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        loss_gx1 = torch.mean((0.9 * torch.ones_like(data_dx1) - data_dx1) ** 2)
        loss_gx2 = torch.mean((0.9 * torch.ones_like(data_dx2) - data_dx2) ** 2)

        return loss + loss_gx1 + loss_gx2



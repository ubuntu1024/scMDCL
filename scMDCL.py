import opt
from encoder import *
import torch


class scMDCL(nn.Module):
    def __init__(self, gae1, gae2, n_node=None):
        super(scMDCL, self).__init__()

        self.gae1 = gae1
        self.gae2 = gae2

        self.cluster_centers1 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        self.cluster_centers2 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers1.data)
        torch.nn.init.xavier_normal_(self.cluster_centers2.data)
        self.q_distribution1 = q_distribution(self.cluster_centers1)
        self.q_distribution2 = q_distribution(self.cluster_centers2)


    
    def FIRND(self, adj, z_igae):
        z_i =  z_igae
        zl = torch.spmm(adj, z_i)
        z_tilde=zl

        return z_tilde

    def forward(self, x1, adj1, x2, adj2, pretrain=False):


        # node embedding encoded by IGAE
        z_igae1, a_igae1 = self.gae1.encoder(x1, adj1)
        z_igae2, a_igae2 = self.gae2.encoder(x2, adj2)



        z1 = self.FIRND(adj1, z_igae1)
        z2 = self.FIRND(adj2, z_igae2)


        cons=[z1,z2]

        # IGAE decoding
        z_hat1, z_adj_hat1 = self.gae1.decoder(z1, adj1)
        a_hat1 = a_igae1 + z_adj_hat1

        z_hat2, z_adj_hat2 = self.gae2.decoder(z2, adj2)
        a_hat2 = a_igae2 + z_adj_hat2

        if not pretrain:
            # the soft assignment distribution Q
            Q1 = self.q_distribution1(z1, z_igae1)
            Q2 = self.q_distribution2(z2, z_igae2)

        else:
            Q1, Q2 = None, None

        return  z_hat1, a_hat1,  z_hat2, a_hat2, Q1, Q2, z1, z2, cons

import argparse

parser = argparse.ArgumentParser(description='scMDCL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
parser.add_argument('--name', type=str, default="Ma-2020-1")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--rec_epoch', type=int, default=50)
parser.add_argument('--fus_epoch', type=int, default=100)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--pretrain', type=bool, default=False)

# parameters
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--alpha1', type=float, default=0.1)
parser.add_argument('--alpha2', type=float, default=10)
parser.add_argument('--method', type=str, default='euc')
parser.add_argument('--second_view', type=str, default='ATAC')#ATAC or ADTS
parser.add_argument('--lr', type=float, default=1e-3)

# dimension of input and latent representations
parser.add_argument('--n_d1', type=int, default=100)
parser.add_argument('--n_d2', type=int, default=100)#ATAC is 100 and ADTS is 209
parser.add_argument('--n_z', type=int, default=20)

# IGAE structure parameter
parser.add_argument('--gae_n_enc_1', type=int, default=256)
parser.add_argument('--gae_n_enc_2', type=int, default=128)
parser.add_argument('--gae_n_dec_1', type=int, default=128)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0)

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--ami', type=float, default=0)

args = parser.parse_args()

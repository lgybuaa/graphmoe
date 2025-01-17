import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--batch', default=512, type=int, help='training batch size')
    parser.add_argument('--tst_batch', default=128, type=int, help='testing batch size')
    parser.add_argument('--inv_fuc', default='irm', type=str, help='Inv penalty fuction')
    parser.add_argument('--lambda_inv', default=0.01, type=float,help='lambda of inv plenty')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
    parser.add_argument('--data_dir', default='/home/luguangyue/lgy/graph_MoE/dataset/mygraph_datasets/', help='data dir')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
    parser.add_argument('--topk', default=3, type=int, help='topk expert to use')
    parser.add_argument('--topk_pred', default=20, type=int, help='topk in evaluation')
    parser.add_argument('--workers', default=0, type=int, help='number of workers in dataloader')
    parser.add_argument('--neg_sampling_ratio', default=1.0, type=float, help='ratio of negative sampling')

    parser.add_argument('--latdim', default=128, type=int, help='latent dimensionality')
    parser.add_argument('--inter_dim', default=256, type=int, help='mlp intermediate dimensionality')
    parser.add_argument('--router_aux_loss_factor', default=0.02, type=float, help='number of mlp layers')
    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
    parser.add_argument('--train_datasets', default='link1', type=str, help='which set of datasets to use')
    parser.add_argument('--test_datasets', default='link2', type=str, help='which set of datasets to use')
    parser.add_argument('--drop_rate', default=0.1, type=float, help='ratio of dropout')
    parser.add_argument('--reca_range', default=0.2, type=float, help='range of recalibration')
    parser.add_argument('--expert_num', default=8, type=int, help='number of experts')
    parser.add_argument('--temperature', default=1, type=float, help='temperature in softmax')
    parser.add_argument('--add_noise', default=False, type=str2bool, help='if add noise to logists')
    parser.add_argument('--noise_mult', default=1., type=float, help='noise level')
    parser.add_argument('--flag_router', default=False, type=str2bool, help='if print router')
    return parser.parse_args()
args = parse_args()
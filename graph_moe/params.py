import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch', default=512, type=int, help='training batch size')
    parser.add_argument('--inv_fuc', default='irm', type=str, help='Inv penalty fuction')
    parser.add_argument('--lambda_inv', default=0.01, type=float,help='lambda of inv plenty')
    parser.add_argument('--tst_batch', default=256, type=int, help='testing batch size (number of users)')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--tst_epoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
    parser.add_argument('--topk', default=20, type=int, help='topk in evaluation')
    parser.add_argument('--epoch_max_step', default=-1, type=int, help='indicates the maximum number of steps in one epoch, -1 denotes full steps')
    parser.add_argument('--trn_mode', default='train-all', type=str, help='[fewshot, train-all]')
    parser.add_argument('--tst_mode', default='tst', type=str, help='[tst, val]')
    parser.add_argument('--eval_loss', default=True, type=bool, help='whether use CE loss to evaluate test performance')
    parser.add_argument('--ratio_fewshot', default=0.1, type=float, help='ratio of fewshot set')
    parser.add_argument('--tst_steps', default=-1, type=int, help='number of test steps, -1 indicates all')
    parser.add_argument('--workers', default=0, type=int, help='number of workers in dataloader')

    parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')
    parser.add_argument('--latdim', default=128, type=int, help='latent dimensionality')
    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
    parser.add_argument('--fc_layer', default=8, type=int, help='number of fully-connected layers')
    parser.add_argument('--gt_layer', default=2, type=int, help='number of graph transformer layers')
    parser.add_argument('--head', default=4, type=int, help='number of attention heads')
    parser.add_argument('--anchor', default=256, type=int, help='number of anchor nodes in the compressed graph transformer')
    parser.add_argument('--act', default='relu', type=str, help='activation function')
    parser.add_argument('--train_datasets', default='link1', type=str, help='which set of datasets to use')
    parser.add_argument('--test_datasets', default='link2', type=str, help='which set of datasets to use')
    parser.add_argument('--scale_layer', default=10, type=float, help='per-layer scale factor')
    parser.add_argument('--leaky', default=0.5, type=float, help='slope of leaky relu activation')
    parser.add_argument('--drop_rate', default=0.1, type=float, help='ratio of dropout')
    parser.add_argument('--reca_range', default=0.2, type=float, help='range of recalibration')
    parser.add_argument('--selfloop', default=0, type=int, help='indicating using self-loop or not')
    parser.add_argument('--niter', default=2, type=int, help='number of iterations in svd')
    parser.add_argument('--expert_num', default=8, type=int, help='number of experts')
    parser.add_argument('--loss', default='ce', type=str, help='loss function')
    parser.add_argument('--proj_method', default='both', type=str, help='feature projection method')
    parser.add_argument('--nn', default='mlp', type=str, help='what trainable network to use')
    parser.add_argument('--proj_trn_steps', default=100, type=int, help='number of training steps for one initial projection')
    parser.add_argument('--attempt_cache', default=10000000, type=int, help='number of training steps for one initial projection')
    parser.add_argument('--temperature', default=1, type=float, help='temperature in softmax')
    parser.add_argument('--add_noise', default=False, type=bool, help='if add noise to logists')
    parser.add_argument('--noise_mult', default=1., type=float, help='noise level')
    return parser.parse_args()
args = parse_args()
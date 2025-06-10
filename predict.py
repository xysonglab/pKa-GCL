import os
import time
import argparse
import numpy as np
import paddle
import paddle.nn.functional as F
from dataloader import DualDataLoader as Dataloader
from dataset import Molecule2DView, Molecule3DView
from model_mean import pKaMPNN
from utils import rmse
from sklearn.metrics import roc_auc_score

def setup_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)

@paddle.no_grad()
def evaluate(model, loader, task, result_dir_fold, flag):
    model.eval()
    y_hat_list = []
    y_list = []
    y_smiles_list = []
    path = os.path.join(result_dir_fold, flag+'_predict.csv')
    with open(path, "w") as f1:
        f1.write("flag,smile,predict,true\n")
        for _ in range(loader.steps):
            graph_2d, graph_3d, smiles, y = loader.next_batch()
            y_hat = model(graph_2d, graph_3d)
            y_hat_list += y_hat.tolist()
            y_list += y.tolist()
            y_smiles_list += smiles

        for t1, t2, t3 in zip(y_smiles_list, y_hat_list, y_list):
            f1.write(f"{flag},{t1},{t2},{t3}\n")

    y_hat = np.array(y_hat_list)
    y = np.array(y_list)
    
    if task == 'regression':
        score = rmse(y, y_hat)
    else:
        auc_score_list = []
        if y.shape[1] > 1:
            for label in range(y.shape[1]):
                true, pred = y[:, label], y_hat[:, label]
                if len(true[np.where(true >= 0)]) == 0:
                    continue
                if len(set(true[np.where(true >= 0)])) == 1:
                    auc_score_list.append(float('nan'))
                else:
                    auc_score_list.append(roc_auc_score(true[np.where(true >= 0)], pred[np.where(true >= 0)]))
            score = np.nanmean(auc_score_list)
        else:
            score = roc_auc_score(y, y_hat)
    return score


def load_best_model(model, optimizer, scheduler, checkpoint_path):
    # Load saved model/optimizer/scheduler states
    checkpoint = paddle.load(checkpoint_path)
    
    # Restore model parameters
    model.set_state_dict(checkpoint['model'])
    
    # Restore optimizer state
    optimizer.set_state_dict(checkpoint['optimizer'])
    
    # Restore scheduler state
    scheduler.set_state_dict(checkpoint['scheduler'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='atte')
    parser.add_argument('--model_dir', type=str, default='./output/bpka_30000/fold_6')
    parser.add_argument('--output_dir', type=str, default='./output/atte')
    parser.add_argument('--results_dir', type=str, default='./results/atte_b')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--num_folds", type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_dec_rate", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop_pool", type=float, default=0.2)
    parser.add_argument("--dec_epoch", type=int, default=10)
    parser.add_argument('--tol_epoch', type=int, default=50)
    parser.add_argument('--all_steps', type=int, default=40000)
    parser.add_argument("--task_dim", type=int, default=1)

    parser.add_argument("--rbf_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_convs", type=int, default=2)
    parser.add_argument("--num_pool", type=int, default=2)
    parser.add_argument("--num_dist", type=int, default=0)
    parser.add_argument("--num_angle", type=int, default=4)
    parser.add_argument('--max_dist_2d', type=float, default=3.0)
    parser.add_argument('--cut_dist', type=float, default=5.0)
    parser.add_argument("--spa_w", type=float, default=0.1)
    parser.add_argument("--mlp_dims", type=str, default='128*4,128*2,128')

    args = parser.parse_args()
    setup_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if int(args.cuda) == -1:
        paddle.set_device('cpu')
    else:
        paddle.set_device(f'gpu:{args.cuda}')

    # Load dataset
    data_2d = Molecule2DView(args.data_dir, args.dataset)
    data_3d = Molecule3DView(args.data_dir, args.dataset, args.cut_dist, args.num_angle)

    # Get dimension values
    node_in_dim = data_2d.atom_feat_dim  # Get 2D graph atom feature dim
    edge_in_dim = data_2d.bond_feat_dim  # Get 2D graph bond feature dim
    atom_in_dim = data_3d.atom_feat_dim  # Get 3D graph atom feature dim
    mlp_dims = [eval(dim) for dim in args.mlp_dims.split(',')]
    print(f"Initializing pKaMPNN with num_dist: {args.num_dist}")
    tst_loader = Dataloader(data_2d, data_3d, args.batch_size, False)
    # Initialize model
    model = pKaMPNN(node_in_dim, edge_in_dim, atom_in_dim, args.rbf_dim, args.hidden_dim, \
                        args.max_dist_2d, args.cut_dist, mlp_dims, args.spa_w, \
                        args.num_convs, args.num_pool, args.num_dist, args.num_angle, \
                        args.dropout, args.drop_pool, args.task_dim, activation=F.relu)
    epoch_step = len(tst_loader)
    boundaries = [i for i in range(args.dec_epoch*epoch_step, args.all_steps, args.dec_epoch*epoch_step)]
    values = [args.lr * args.lr_dec_rate ** i for i in range(0, len(boundaries) + 1)]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values, verbose=False)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters(), weight_decay=args.weight_decay)
    # Load pretrained weights
    checkpoint_path = os.path.join(args.model_dir, 'saved_model')

    model = load_best_model(model, optimizer, scheduler, checkpoint_path)

    # Load test data
    
    result_dir_fold = args.results_dir + '/test'
    if not os.path.isdir(result_dir_fold):
        os.makedirs(result_dir_fold)

    # Evaluate model
    test_score = evaluate(model, tst_loader, task='regression', result_dir_fold=result_dir_fold, flag="test")
    print(f"测试集评分: {test_score}")

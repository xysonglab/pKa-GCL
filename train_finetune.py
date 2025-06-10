import os
import time
import math
import argparse
import random
import numpy as np

import paddle
import paddle.nn.functional as F
from dataloader import DualDataLoader as Dataloader
from dataset import Molecule2DView, Molecule3DView
from model_mean import pKaMPNN
from utils import split_dataset, rmse
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def setup_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@paddle.no_grad()
def evaluate(model, loader, task, result_dir_fold,flag ):
    model.eval()
    y_hat_list = []
    y_list = []
    y_smiles_list = []
    path = os.path.join(result_dir_fold, flag+'_predict.csv')
    f1=open(path,"w")
    f1.write("flag,smile,predict,true\n")
    for _ in range(loader.steps):
        graph_2d, graph_3d,smiles, y = loader.next_batch()
        y_hat = model(graph_2d, graph_3d)
        y_hat_list += y_hat.tolist()
        y_list += y.tolist()
        y_smiles_list+=smiles
    for t1,t2,t3 in zip(y_smiles_list,y_hat_list,y_list):
        f1.write(flag+","+t1+","+str(t2)+","+str(t3)+"\n")

    y_hat = np.array(y_hat_list)
    y = np.array(y_list)
    if task == 'regression':
        score = rmse(y, y_hat)
    else:
        auc_score_list = []
        if y.shape[1] > 1:
            for label in range(y.shape[1]):
                true, pred = y[:, label], y_hat[:, label]
                # all 0's or all 1's
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



def train(args, model, trn_loader, tst_loader ,val_loader, output_dir_fold,result_dir_fold):
    epoch_step = len(trn_loader)
    boundaries = [i for i in range(args.dec_epoch*epoch_step, args.all_steps, args.dec_epoch*epoch_step)]
    values = [args.lr * args.lr_dec_rate ** i for i in range(0, len(boundaries) + 1)]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values, verbose=False)
    optim = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters(), weight_decay=args.weight_decay)

    best_score = 1e9 if args.task == 'regression' else 0
    best_score_val = best_score
    running_log, best_epoch = '', 0
    print('Start training model...')
    loss_path = os.path.join(result_dir_fold, 'train_loss.csv')
    f_loss = open(loss_path, "w")
    f_loss.write("epoch,loss\n")
    for epoch in range(1, int(args.all_steps/epoch_step) + 1):
        sum_loss = 0.
        model.train()
        start = time.time()
        y_hat_list = []
        y_list = []
        y_smiles_list = []
        for _ in range(trn_loader.steps):
            graph_2d, graph_3d,smiles,  y = trn_loader.next_batch()
            y_hat = model(graph_2d, graph_3d)

            y_hat_list += y_hat.tolist()
            y_list += y.tolist()
            y_smiles_list += smiles
            if args.task == 'regression':
                loss = model.loss_reg(y_hat, y)
            else:
                loss = model.loss_cls(y_hat, y)
            loss.backward()
            optim.step()
            optim.clear_grad()
            scheduler.step()
            sum_loss += loss
        
        trn_loss = sum_loss/(len(trn_loader))
        end_trn = time.time()
        val_score = evaluate(model, val_loader, args.task,result_dir_fold,flag="val")
        tst_score = evaluate(model, tst_loader, args.task,result_dir_fold,flag="test")
        end_val = time.time()

        if args.task == 'regression':
            metric = 'rmse'
            if val_score < best_score_val:
                best_score_val = val_score
                best_score = tst_score
                best_epoch = epoch
                path = os.path.join(result_dir_fold, 'train_predict.csv')
                f1 = open(path, "w")
                f1.write("flag,smile,predict,true\n")
                for t1, t2, t3 in zip(y_smiles_list, y_hat_list, y_list):
                    f1.write("train"+","+t1 + "," + str(t2) + "," + str(t3) + "\n")
                f1.close()

                # Persist model and training states
                obj = {
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),  # Store optimizer parameters
                    'scheduler': scheduler.state_dict(),  # Maintain scheduler configuration
                    'epoch': epoch
                }
                path = os.path.join(output_dir_fold, 'saved_model')
                paddle.save(obj, path)
        else:
            metric = 'auc'
            if val_score > best_score_val:
                best_score_val = val_score
                best_score = tst_score
                best_epoch = epoch
                # Archive complete training context
                obj = {
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),  # Preserve optimizer status
                    'scheduler': scheduler.state_dict(),  # Retain scheduler settings
                    'epoch': epoch
                }
                path = os.path.join(output_dir_fold, 'saved_model')
                paddle.save(obj, path)

        log = f'Epoch: %d/{int(args.all_steps/epoch_step)}, loss: %.4f, Val {metric}: %.6f, Test {metric}: %.6f,' % (epoch, trn_loss, val_score, tst_score)
        log += f'Best Test {metric}: %.6f, time: %.2f, val_time: %.2f.\n' % (best_score, end_trn-start, end_val-end_trn)
        f_loss.write(str(epoch)+","+str(trn_loss)+"\n")
        print(log)
        running_log += log
        f = open(os.path.join(output_dir_fold, 'log.txt'), 'w')
        f.write(running_log)
        f.close()

        if epoch - best_epoch > args.tol_epoch:
            break

    running_log += f'The best epoch is %d, best val {metric} is %.6f, test {metric} is %.6f. \n' % (best_epoch, best_score_val, best_score)
    f = open(os.path.join(output_dir_fold, 'log.txt'), 'w')
    f.write(running_log)
    f.close()
    return best_score

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='pka_30000')
    parser.add_argument('--model_dir', type=str, default='./runs/pka_30000')
    parser.add_argument('--output_dir', type=str, default='./output/pka_30000')
    parser.add_argument('--results_dir', type=str, default='./results/pka_30000')
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
    parser.add_argument('--max_dist_2d', type=float, default=3.)
    parser.add_argument('--cut_dist', type=float, default=5.)
    parser.add_argument("--spa_w", type=float, default=0.1)
    parser.add_argument("--mlp_dims", type=str, default='128*4,128*2,128')

    args = parser.parse_args()
    setup_seed(args.seed)
    if not args.num_dist:
        args.num_dist = 2 if args.cut_dist <= 4 else 4
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if int(args.cuda) == -1:
        paddle.set_device('cpu')
    else:
        paddle.set_device('gpu:%s' % args.cuda)
    
    if args.dataset in ['pka_30000', 'bpka_30000']:
        args.task = 'regression'
    else:
        print('The dataset %s is not included.' % args.dataset)
        exit(-1)
    
    data_2d = Molecule2DView(args.data_dir, args.dataset)
    data_3d = Molecule3DView(args.data_dir, args.dataset, args.cut_dist, args.num_angle, args.num_dist)
    print('Dataset size,', len(data_2d))
    print(data_2d.smile_list)
    assert len(data_2d) == len(data_3d)
    node_in_dim = data_2d.atom_feat_dim
    edge_in_dim = data_2d.bond_feat_dim
    atom_in_dim = data_3d.atom_feat_dim
    mlp_dims = [eval(dim) for dim in args.mlp_dims.split(',')]

    scores = []
    for idx in range(args.num_folds):
        trn_data_2d, tst_data_2d, val_data_2d = split_dataset(data_2d, idx, args.num_folds)
        print("*************************************")
        print(len(trn_data_2d))
        print(len(tst_data_2d))
        print(len(val_data_2d))
        print("*************************************")
        trn_data_3d, tst_data_3d, val_data_3d = split_dataset(data_3d, idx, args.num_folds)
        trn_loader = Dataloader(trn_data_2d, trn_data_3d, args.batch_size, True)
        tst_loader = Dataloader(tst_data_2d, tst_data_3d, args.batch_size, False)
        val_loader = Dataloader(val_data_2d, val_data_3d, args.batch_size, False)

        output_dir_fold = args.output_dir + '/fold_' + str(idx)
        result_dir_fold = args.results_dir + '/fold_' + str(idx)
        if not os.path.isdir(output_dir_fold):
            os.mkdir(output_dir_fold)
        if not os.path.isdir(result_dir_fold):
            os.makedirs(result_dir_fold)
        model = pKaMPNN(node_in_dim, edge_in_dim, atom_in_dim, args.rbf_dim, args.hidden_dim, \
                        args.max_dist_2d, args.cut_dist, mlp_dims, args.spa_w, \
                        args.num_convs, args.num_pool, args.num_dist, args.num_angle, \
                        args.dropout, args.drop_pool, args.task_dim, activation=F.relu)

        if os.path.isdir(args.model_dir):
            path = os.path.join(args.model_dir, 'saved_model_10_0.900000')
            load_state_dict = paddle.load(path)
            model_state_dict = model.state_dict()
            for k, v in load_state_dict['model'].items():
                if 'proj_' in k:
                    continue
                model_state_dict[k] = v
            model.set_state_dict(model_state_dict)
            print('Load pre-trained model from %s at epoch %d.' % (path, load_state_dict['epoch']))
        else:
            print('Not load pre-trained model.')

        score = train(args, model, trn_loader, tst_loader, val_loader, output_dir_fold,result_dir_fold)
        scores.append(score)
    
    f = open(os.path.join(args.output_dir, 'scores.txt'), 'w')
    log = str(scores) + '\n' + str(np.mean(scores))
    f.write(log)
    f.close()

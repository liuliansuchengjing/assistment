# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:42:32 2021

@author: Ling Sun
"""

import argparse
import time
import numpy as np
import Constants
import torch
import torch.nn as nn
from graphConstruct import ConRelationGraph, ConHyperGraphList
from dataLoader import Split_data, DataLoader
from Metrics_schemeA_openloop import Metrics, KTLoss
from HGAT import MSHGAT
from Optim import ScheduledOptim


torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

metric = Metrics()

parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='Assist')
parser.add_argument('-epoch', type=int, default=120)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-d_model', type=int, default=64)
parser.add_argument('-initialFeatureSize', type=int, default=64)
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.3)
parser.add_argument('-log', default=None)
parser.add_argument('-save_path', default="./checkpoint/DiffusionPrediction_a150.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-pos_emb', type=bool, default=True)

opt = parser.parse_args()
opt.d_word_vec = opt.d_model


# print(opt)


def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


def train_epoch(model, training_data, graph, hypergraph_list, loss_func, kt_loss, optimizer):
    # train

    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    batch_num = 0.0
    auc_train = []
    acc_train = []
    #新增指标
    kt_prec_train = []
    kt_rec_train = []
    kt_f1_train = []


    for i, batch in enumerate(
            training_data):  # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # data preparing
        tgt, tgt_timestamp, tgt_idx, ans = (item.cuda() for item in batch)

        np.set_printoptions(threshold=np.inf)
        gold = tgt[:, 1:]

        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        # training
        optimizer.zero_grad()
        # pred= model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
        pred, pred_res, kt_mask = model(tgt, tgt_timestamp, tgt_idx, ans, graph,
                                        hypergraph_list)  # ==================================

        # loss
        loss, n_correct = get_performance(loss_func, pred, gold)
        # loss_kt, auc, acc = kt_loss(pred_res, ans,
        #                             kt_mask)  # ============================================================================
        loss_kt, auc, acc, kt_prec, kt_rec, kt_f1 = kt_loss(pred_res, ans, kt_mask)

        # loss = loss + loss_kt
        loss = loss
        # print("loss_kt:", loss_kt)

        loss.backward()

        # parameter update
        optimizer.step()
        optimizer.update_learning_rate()

        n_total_correct += n_correct
        total_loss += loss.item()
        # if auc != -1 and acc != -1:  # ========================================================================================
        #     auc_train.append(
        #         auc)  # ====================================================================================
        #     acc_train.append(
        #         acc)  # ==========================================================================================
        if auc != -1 and acc != -1:
            auc_train.append(auc)
            acc_train.append(acc)
            kt_prec_train.append(kt_prec)
            kt_rec_train.append(kt_rec)
            kt_f1_train.append(kt_f1)


    # return total_loss / n_total_words, n_total_correct / n_total_words, auc_train, acc_train
    return total_loss / n_total_words, n_total_correct / n_total_words, auc_train, acc_train, kt_prec_train, kt_rec_train, kt_f1_train

def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else -1.0


def train_model(MSHGAT, data_path):
    # ========= Preparing DataLoader =========#
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    train_data = DataLoader(train, batch_size=opt.batch_size, load_dict=True, cuda=False)
    valid_data = DataLoader(valid, batch_size=opt.batch_size, load_dict=True, cuda=False)
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    opt.user_size = user_size

    # ========= Preparing Model =========#
    model = MSHGAT(user_size, opt, dropout=opt.dropout)
    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
    kt_loss = KTLoss()

    params = model.parameters()
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()
        kt_loss = kt_loss.cuda()

    validation_history = 0.0
    best_scores = {}
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')

        start = time.time()
        # train_loss, train_accu, train_auc, train_acc = train_epoch(model, train_data, relation_graph, hypergraph_list,
        #                                                            loss_func, kt_loss, optimizer)
        train_loss, train_accu, train_auc, train_acc, train_kt_p, train_kt_r, train_kt_f1 = train_epoch(model, train_data, relation_graph, hypergraph_list,
                                                                   loss_func, kt_loss, optimizer)


        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))
        print('auc_test: {:.10f}'.format(np.mean(train_auc)),
              'acc_test: {:.10f}'.format(np.mean(train_acc)))
        print('KT auc: {:.6f} acc: {:.6f} P: {:.6f} R: {:.6f} F1: {:.6f}'.format(
            safe_mean(train_auc), safe_mean(train_acc),
            safe_mean(train_kt_p), safe_mean(train_kt_r), safe_mean(train_kt_f1)
        ))

        if epoch_i >= 0:
            start = time.time()
            # scores, auc_test, acc_test = test_epoch(model, valid_data, relation_graph, hypergraph_list, kt_loss)
            scores, auc_test, acc_test, kt_p, kt_r, kt_f1 = test_epoch(model, valid_data, relation_graph, hypergraph_list, kt_loss)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print('auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)))
            print("Validation use time: ", (time.time() - start) / 60, "min")

            print('  - (Test) ')
            scores, auc_test, acc_test, kt_p, kt_r, kt_f1 = test_epoch(model, test_data, relation_graph, hypergraph_list, kt_loss)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print('auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)))
            val_score = scores.get("acc@20", sum(scores.values()))
            if validation_history <= val_score:
                print("Best Validation acc@20:{} at Epoch:{}".format(scores.get("acc@20", 0.0), epoch_i))
                validation_history = val_score
                best_scores = scores
                print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)

    print(" -(Finished!!) \n Best scores: ")
    for metric in best_scores.keys():
        print(metric + ' ' + str(best_scores[metric]))


def test_epoch(model, validation_data, graph, hypergraph_list, kt_loss, k_list=[5, 10, 20],
               mode: str = "teacher_forcing", max_starts_per_seq=None):
    """Evaluation phase.

    Two modes:
      1) teacher_forcing (default): current behavior (windows over per-position argmax from a single forward pass)
      2) open_loop: *one-shot planning* rollout.
         For each start position t, we mask the future and iteratively feed back the predicted top-1 token to generate a k-step path.

    Scheme A (both modes): for a given k, only evaluate windows where future ground-truth length >= k.

    Args:
        max_starts_per_seq: cap the number of start positions per sequence (to control runtime).
            - None: use all valid starts.
            - int: uniformly subsample that many starts from each sequence.
    """
    model.eval()
    auc_test = []
    acc_test = []
    kt_prec_test = []
    kt_rec_test = []
    kt_f1_test = []

    # init score accumulators
    scores = {}
    for k in k_list:
        scores[f'acc@{k}'] = 0.0
        scores[f'exact@{k}'] = 0.0
        scores[f'precision@{k}'] = 0.0
        scores[f'recall@{k}'] = 0.0
        scores[f'f1@{k}'] = 0.0
        scores[f'map@{k}'] = 0.0
        scores[f'NDCG@{k}'] = 0.0

    # count evaluated windows per k (Scheme A differs across k)
    win_cnt = {k: 0 for k in k_list}

    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            tgt, tgt_timestamp, tgt_idx, ans = batch

            # gold sequence (B, T)
            gold_seq = tgt[:, 1:].detach().cpu().numpy()
            B = tgt.size(0)
            T = gold_seq.shape[1]

            # --- KT metrics computed once per batch (teacher-forcing forward) ---
            pred_full, pred_res, kt_mask = model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
            loss_kt, auc, acc, kt_prec, kt_rec, kt_f1 = kt_loss(pred_res.cpu(), ans.cpu(), kt_mask.cpu())
            if auc != -1 and acc != -1:
                auc_test.append(auc)
                acc_test.append(acc)
                kt_prec_test.append(kt_prec)
                kt_rec_test.append(kt_rec)
                kt_f1_test.append(kt_f1)

            if mode == "teacher_forcing":
                # pred_full: (B*T, V) -> (B, T, V)
                pred_seq = pred_full.view(B, T, pred_full.size(-1)).detach().cpu().numpy()
                scores_batch, win_len = metric.compute_path_metric(pred_seq, gold_seq, k_list)
                if win_len == 0:
                    continue
                for k in k_list:
                    # same win_len weighting as your original implementation
                    scores[f'acc@{k}'] += scores_batch[f'acc@{k}'] * win_len
                    scores[f'exact@{k}'] += scores_batch[f'exact@{k}'] * win_len
                    scores[f'precision@{k}'] += scores_batch[f'precision@{k}'] * win_len
                    scores[f'recall@{k}'] += scores_batch[f'recall@{k}'] * win_len
                    scores[f'f1@{k}'] += scores_batch[f'f1@{k}'] * win_len
                    scores[f'map@{k}'] += scores_batch[f'map@{k}'] * win_len
                    scores[f'NDCG@{k}'] += scores_batch[f'NDCG@{k}'] * win_len
                    win_cnt[k] += win_len  # same total across k here
                continue

            if mode != "open_loop":
                raise ValueError(f"Unknown mode={mode}. Use 'teacher_forcing' or 'open_loop'.")

            # --- open-loop rollout: one-shot planning at each start ---
            max_k = max(k_list)
            min_k = min(k_list)

            for b in range(B):
                gt_row = gold_seq[b]
                valid_mask = (gt_row != Constants.PAD) & (gt_row != 1)
                valid_len = int(valid_mask.sum())
                if valid_len <= 0:
                    continue

                # valid start positions in gold_seq coordinates (0..valid_len-1)
                starts = list(range(0, valid_len))
                # require at least min_k future from start
                starts = [s for s in starts if (valid_len - s) >= min_k]
                if not starts:
                    continue

                # optionally subsample starts to control runtime
                if max_starts_per_seq is not None and len(starts) > max_starts_per_seq:
                    # uniform subsample
                    idxs = np.linspace(0, len(starts) - 1, num=max_starts_per_seq, dtype=int).tolist()
                    starts = [starts[j] for j in idxs]

                for start in starts:
                    horizon = min(max_k, valid_len - start)
                    if horizon < min_k:
                        continue

                    # build masked prefix-only inputs for this start
                    tgt_w = tgt[b:b+1].clone()
                    ts_w = tgt_timestamp[b:b+1].clone()
                    ans_w = ans[b:b+1].clone()
                    idx_w = tgt_idx[b:b+1].clone()

                    # mask future beyond start in tgt (note: tgt index is shifted by +1 vs gold_seq)
                    cut = start + 1  # keep tgt[:, :cut+1], mask tgt[:, cut+1:]
                    if cut + 1 < tgt_w.size(1):
                        tgt_w[:, cut + 1:] = Constants.PAD

                        # timestamps / answers may be (B,L) or something else; only mask if sequence-shaped
                        if ts_w.dim() == 2 and ts_w.size(1) == tgt_w.size(1):
                            ts_w[:, cut + 1:] = 0
                        if ans_w.dim() == 2 and ans_w.size(1) == tgt_w.size(1):
                            ans_w[:, cut + 1:] = 0
                        # tgt_idx is often (B,) (user id) -> do not mask unless sequence-shaped
                        if idx_w.dim() == 2 and idx_w.size(1) == tgt_w.size(1):
                            idx_w[:, cut + 1:] = 0

                    pred_path = []
                    for step in range(horizon):
                        pred_step, _, _ = model(tgt_w, ts_w, idx_w, ans_w, graph, hypergraph_list)
                        # reshape to (1, T, V); position 'start+step' predicts tgt token at (start+step+1)
                        pred_step_seq = pred_step.view(1, T, pred_step.size(-1))
                        next_token = int(torch.argmax(pred_step_seq[0, start + step, :]).item())

                        # write back prediction as the next interaction token
                        tgt_pos = (start + step + 1) + 0  # in tgt space
                        if tgt_pos < tgt_w.size(1):
                            tgt_w[0, tgt_pos] = next_token
                        pred_path.append(next_token)

                    # score for each k that fits the horizon (Scheme A)
                    for k in k_list:
                        if k > horizon:
                            continue
                        gt_win = gt_row[start:start + k].tolist()
                        pred_win = pred_path[:k]
                        wm = metric.window_metrics(pred_win, gt_win)

                        scores[f'acc@{k}'] += wm['pos_acc']
                        scores[f'exact@{k}'] += wm['exact']
                        scores[f'precision@{k}'] += wm['precision']
                        scores[f'recall@{k}'] += wm['recall']
                        scores[f'f1@{k}'] += wm['f1']
                        scores[f'map@{k}'] += wm['map']
                        scores[f'NDCG@{k}'] += wm['ndcg']
                        win_cnt[k] += 1

    # normalize
    for k in k_list:
        denom = win_cnt[k]
        if denom > 0:
            scores[f'acc@{k}'] /= denom
            scores[f'exact@{k}'] /= denom
            scores[f'precision@{k}'] /= denom
            scores[f'recall@{k}'] /= denom
            scores[f'f1@{k}'] /= denom
            scores[f'map@{k}'] /= denom
            scores[f'NDCG@{k}'] /= denom
        else:
            # keep zeros
            pass

    return scores, auc_test, acc_test, kt_prec_test, kt_rec_test, kt_f1_test

def test_model(MSHGAT, data_path, mode="teacher_forcing", max_starts_per_seq=None):
    kt_loss = KTLoss()
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    opt.user_size = user_size

    model = MSHGAT(user_size, opt, dropout=opt.dropout)
    model.load_state_dict(torch.load(opt.save_path))
    model.cuda()
    kt_loss = kt_loss.cuda()
    scores, auc_test, acc_test, kt_p, kt_r, kt_f1 = test_epoch(model, test_data, relation_graph, hypergraph_list, kt_loss, k_list=[1,3, 5, 10], mode=mode, max_starts_per_seq=max_starts_per_seq)
    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))
    print('auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)))
    print('KT auc: {:.6f} acc: {:.6f} P: {:.6f} R: {:.6f} F1: {:.6f}'.format(
        safe_mean(auc_test), safe_mean(acc_test),
        safe_mean(kt_p), safe_mean(kt_r), safe_mean(kt_f1)
    ))


if __name__ == "__main__":
    model = MSHGAT
    # train_model(model, opt.data_name)
    # test_model(model, opt.data_name, mode="open_loop", max_starts_per_seq=50)
    test_model(model, opt.data_name, mode="open_loop", max_starts_per_seq=50)
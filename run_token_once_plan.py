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
from Metrics_token_path import Metrics, KTLoss
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
            if validation_history <= sum(scores.values()):
                print("Best Validation hit@100:{} at Epoch:{}".format(scores["hits@20"], epoch_i))
                validation_history = sum(scores.values())
                best_scores = scores
                print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)

    print(" -(Finished!!) \n Best scores: ")
    for metric in best_scores.keys():
        print(metric + ' ' + str(best_scores[metric]))


def test_epoch(model, validation_data, graph, hypergraph_list, kt_loss, k_list=[5, 10, 20]):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    auc_test = []
    acc_test = []
    kt_prec_test = []
    kt_rec_test = []
    kt_f1_test = []

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0
        scores['NDCG@' + str(k)] = 0
        scores['precision@' + str(k)] = 0
        scores['recall@' + str(k)] = 0
        scores['f1@' + str(k)] = 0

    n_total_words = 0

    with torch.no_grad():
        for i, batch in enumerate(
                validation_data):  # tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            # print("Validation batch ", i)
            # prepare data
            # tgt, tgt_timestamp, tgt_idx = batch
            tgt, tgt_timestamp, tgt_idx, ans = batch
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

            # forward
            # pred = model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
            pred, pred_res, kt_mask = model(tgt, tgt_timestamp, tgt_idx, ans, graph,
                                            hypergraph_list)  # ==================================
            y_pred = pred.detach().cpu().numpy()
            # loss_kt, auc, acc = kt_loss(pred_res.cpu(), ans.cpu(),
            #                             kt_mask.cpu())  # ====================================================================
            loss_kt, auc, acc, kt_prec, kt_rec, kt_f1 = kt_loss(pred_res.cpu(), ans.cpu(), kt_mask.cpu())

            if auc != -1 and acc != -1:
                auc_test.append(auc)
                acc_test.append(acc)
                kt_prec_test.append(kt_prec)
                kt_rec_test.append(kt_rec)
                kt_f1_test.append(kt_f1)

            scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len
                scores['NDCG@' + str(k)] += scores_batch['NDCG@' + str(k)] * scores_len
                scores['precision@' + str(k)] += scores_batch['precision@' + str(k)] * scores_len
                scores['recall@' + str(k)] += scores_batch['recall@' + str(k)] * scores_len
                scores['f1@' + str(k)] += scores_batch['f1@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] /= n_total_words
        scores['map@' + str(k)] /= n_total_words
        scores['NDCG@' + str(k)] /= n_total_words
        scores['precision@' + str(k)] /= n_total_words
        scores['recall@' + str(k)] /= n_total_words
        scores['f1@' + str(k)] /= n_total_words

    return scores, auc_test, acc_test, kt_prec_test, kt_rec_test, kt_f1_test




def _get_last_valid_index(seq, pad_id=Constants.PAD):
    """Return the last index with non-PAD, assuming PADs are trailing."""
    # seq: (L,)
    nonpad = (seq != pad_id).nonzero(as_tuple=False).view(-1)
    if nonpad.numel() == 0:
        return -1
    return int(nonpad.max().item())


def test_epoch_once_plan_token(model, validation_data, graph, hypergraph_list,
                               m_list=(3, 5, 7, 9),
                               max_starts_per_seq=None,
                               ignore_ids=(Constants.PAD, 1)):
    """Once-plan (open-loop) path evaluation with Scheme A + token-level metrics.

    For each sequence and each start time t:
      - Only history up to t is visible (positions > t are masked).
      - Autoregressively generate a length-m path (greedy argmax rollout).
      - Compare predicted tokens vs ground-truth next m tokens (token-level).
      - Scheme A: if there are not enough future interactions (any PAD in y_true window), skip.

    Returns:
        dict mapping each m to token-level metrics:
          acc@{m}, micro_p@{m}, micro_r@{m}, micro_f1@{m}, macro_p@{m}, macro_r@{m}, macro_f1@{m}
    """
    model.eval()
    device = next(model.parameters()).device

    # accumulate token pairs for each m
    y_true_all = {m: [] for m in m_list}
    y_pred_all = {m: [] for m in m_list}
    windows_used = {m: 0 for m in m_list}

    with torch.no_grad():
        for batch in validation_data:
            tgt, tgt_timestamp, tgt_idx, ans = batch
            tgt = tgt.to(device)
            tgt_timestamp = tgt_timestamp.to(device)
            tgt_idx = tgt_idx.to(device)
            ans = ans.to(device)

            B, L = tgt.size()
            for b in range(B):
                seq = tgt[b]
                last_idx = _get_last_valid_index(seq, pad_id=Constants.PAD)
                if last_idx < 1:
                    continue

                # valid start positions (t) are those where we have at least max(m_list) future tokens
                # We decide per m inside loop.
                for m in m_list:
                    max_t = last_idx - m
                    if max_t < 0:
                        continue

                    starts = list(range(0, max_t + 1))
                    if max_starts_per_seq is not None and len(starts) > max_starts_per_seq:
                        # uniform subsample for fairness
                        idxs = np.linspace(0, len(starts) - 1, num=max_starts_per_seq, dtype=int).tolist()
                        starts = [starts[i] for i in idxs]

                    for t in starts:
                        # ground-truth window: positions t+1 ... t+m
                        y_true_win = tgt[b, t + 1:t + m + 1].detach().cpu().tolist()

                        # Scheme A: if not enough future or contains ignored token -> skip
                        if any((yy in ignore_ids) for yy in y_true_win):
                            continue

                        # Build once-plan masked inputs (only up to t visible)
                        tgt_w = tgt[b:b + 1].clone()
                        ts_w = tgt_timestamp[b:b + 1].clone()
                        idx_w = tgt_idx[b:b + 1].clone()
                        ans_w = ans[b:b + 1].clone()

                        if t + 1 < L:
                            tgt_w[:, t + 1:] = Constants.PAD

                            if ts_w.dim() == 2 and ts_w.size(1) == L:
                                ts_w[:, t + 1:] = 0
                            elif ts_w.dim() == 1 and ts_w.numel() == L:
                                ts_w[t + 1:] = 0

                            # tgt_idx may be (B,) not (B,L) -> only mask when sequence-shaped
                            if idx_w.dim() == 2 and idx_w.size(1) == L:
                                idx_w[:, t + 1:] = 0

                            if ans_w.dim() == 2 and ans_w.size(1) == L:
                                ans_w[:, t + 1:] = 0
                            elif ans_w.dim() == 1 and ans_w.numel() == L:
                                ans_w[t + 1:] = 0

                        # open-loop greedy rollout for m steps
                        y_pred_win = []
                        for step in range(m):
                            pred_logits, _, _ = model(tgt_w, ts_w, idx_w, ans_w, graph, hypergraph_list)
                            row = t + step
                            if row < 0 or row >= pred_logits.size(0):
                                # safety: should not happen
                                y_pred_win = None
                                break
                            next_item = int(torch.argmax(pred_logits[row]).item())
                            y_pred_win.append(next_item)

                            # write prediction into next position
                            pos = t + step + 1
                            if pos < L:
                                tgt_w[0, pos] = next_item
                            else:
                                # safety
                                y_pred_win = None
                                break

                        if y_pred_win is None or len(y_pred_win) != m:
                            continue

                        # collect tokens
                        y_true_all[m].extend(y_true_win)
                        y_pred_all[m].extend(y_pred_win)
                        windows_used[m] += 1

    # compute token-level metrics for each m
    results = {}
    for m in m_list:
        met = metric.compute_token_metrics(y_true_all[m], y_pred_all[m], ignore_ids=ignore_ids)
        results[f'acc@{m}'] = met['acc']
        results[f'micro_p@{m}'] = met['micro_p']
        results[f'micro_r@{m}'] = met['micro_r']
        results[f'micro_f1@{m}'] = met['micro_f1']
        results[f'macro_p@{m}'] = met['macro_p']
        results[f'macro_r@{m}'] = met['macro_r']
        results[f'macro_f1@{m}'] = met['macro_f1']
        results[f'windows@{m}'] = windows_used[m]

    return results



def test_model(MSHGAT, data_path):
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
    scores, auc_test, acc_test, kt_p, kt_r, kt_f1 = test_epoch(model, test_data, relation_graph, hypergraph_list, kt_loss, k_list=[1,3, 5, 10])
    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))
    print('auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)))
    print('KT auc: {:.6f} acc: {:.6f} P: {:.6f} R: {:.6f} F1: {:.6f}'.format(
        safe_mean(auc_test), safe_mean(acc_test),
        safe_mean(kt_p), safe_mean(kt_r), safe_mean(kt_f1)
    ))



    # Once-plan (open-loop) + Scheme A + token-level metrics for path lengths m=3/5/7/9
    once_plan_results = test_epoch_once_plan_token(
        model, test_data, relation_graph, hypergraph_list,
        m_list=(3, 5, 7, 9),
        max_starts_per_seq=None  # set None to evaluate all possible starts (may be slow)
    )
    print('  - (Once-Plan Token Metrics, Scheme A) ')
    for k in sorted(once_plan_results.keys(), key=lambda x: (int(x.split('@')[-1]) if '@' in x else 0, x)):
        print(k + ' ' + str(once_plan_results[k]))


if __name__ == "__main__":
    model = MSHGAT
    # train_model(model, opt.data_name)
    # test_model(model, opt.data_name)
    test_model(model, opt.data_name)

import numpy as np
import json
import pickle
import csv
import random
from collections import defaultdict
import networkx as nx
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score



def load_idx2u():
    with open('/kaggle/working/GCN/data/r_MOOC10000/idx2u.pickle', 'rb') as f:
        return pickle.load(f)


def load_u2idx():
    with open('/kaggle/working/GCN/data/r_MOOC10000/u2idx.pickle', 'rb') as f:
        return pickle.load(f)


def load_course_video():
    data = {}
    with open('/kaggle/input/riginmooccube/MOOCCube/relations/course-video.json', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) == 2:
                course_id, video_id = row
                if course_id not in data:
                    data[course_id] = []
                data[course_id].append(video_id)
    return data


def load_course():
    courses = []
    with open('/kaggle/input/riginmooccube/MOOCCube/entities/course.json', 'r', encoding='utf-8') as f:
        data = f.read()
        start = 0
        end = 0
        while True:
            start = data.find('{', end)
            if start == - 1:
                break
            end = data.find('}', start) + 1
            json_obj = data[start:end]
            try:
                course = json.loads(json_obj)
                courses.append(course)
            except json.decoder.JSONDecodeError as e:
                print(f"解析错误: {e}")
    return courses


class Metrics(object):

    def __init__(self):
        super().__init__()
        self.PAD = 0

    def apk(self, actual, predicted, k=10):
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # if not actual:
        # 	return 0.0
        return score / min(len(actual), k)

    def ndcg(self, y_, topk, k):
        DCG_score = 0
        IDCG_score = 0
        NDCG = 0
        for i in range(k):
            if topk[i] == y_:
                DCG_score += 1 / np.log2(i + 2)
                IDCG_score += 1 / np.log2(2)

        if IDCG_score != 0:
            NDCG = DCG_score / IDCG_score

        return NDCG

    # def compute_metric(self, y_prob, y_true, k_list=[10, 50, 100]):
    #     '''
    #         y_true: (#samples, )
    #         y_pred: (#samples, #users)
    #     '''
    #     scores_len = 0
    #     y_prob = np.array(y_prob)
    #     y_true = np.array(y_true)
    #
    #     scores = {'hits@' + str(k): [] for k in k_list}
    #     scores.update({'map@' + str(k): [] for k in k_list})
    #     for p_, y_ in zip(y_prob, y_true):
    #         if y_ != self.PAD:
    #             scores_len += 1.0
    #             p_sort = p_.argsort()
    #             for k in k_list:
    #                 topk = p_sort[-k:][::-1]
    #                 scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
    #                 scores['map@' + str(k)].extend([self.apk([y_], topk, k)])
    #
    #     scores = {k: np.mean(v) for k, v in scores.items()}
    #     return scores, scores_len

    def compute_metric(self, y_prob, y_true, k_list=[3, 5, 7, 9]):
        '''
            y_true: (#samples, )
            y_pred: (#samples, #users)
        '''
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        scores.update({'NDCG@' + str(k): [] for k in k_list})
        scores.update({'precision@' + str(k): [] for k in k_list})
        scores.update({'recall@' + str(k): [] for k in k_list})
        scores.update({'f1@' + str(k): [] for k in k_list})

        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD and y_ != 1:
                scores_len += 1.0
                p_sort = p_.argsort()
                for k in k_list:
                    topk = p_sort[-k:][::-1]
                    hit = 1.0 if y_ in topk else 0.0
                    scores['hits@' + str(k)].extend([hit])
                    scores['map@' + str(k)].extend([self.apk([y_], topk, k)])
                    scores['NDCG@' + str(k)].extend([self.ndcg(y_, topk, k)])

                    # ---- TopK Precision/Recall/F1 (single ground-truth) ----
                    prec_k = hit / float(k)
                    rec_k = hit  # 因为只有 1 个相关项
                    f1_k = 0.0 if (prec_k + rec_k) == 0 else 2 * prec_k * rec_k / (prec_k + rec_k)

                    scores['precision@' + str(k)].extend([prec_k])
                    scores['recall@' + str(k)].extend([rec_k])
                    scores['f1@' + str(k)].extend([f1_k])

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len

    def window_metrics(self, pred_win, gt_win):
        """Compute metrics for a single k-step window.

        Args:
            pred_win: list[int] length k
            gt_win: list[int] length k
        Returns:
            dict with keys: pos_acc, exact, precision, recall, f1, map, ndcg
        Notes:
            - precision/recall/f1 are computed by multiset overlap (so when |GT|=k, precision==recall).
            - map/ndcg treat gt_win as a relevant set (order-agnostic relevance).
        """
        from collections import Counter

        k = len(gt_win)
        if k == 0:
            return {
                'pos_acc': 0.0, 'exact': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'map': 0.0, 'ndcg': 0.0
            }

        # order-sensitive
        pos_acc = float(np.mean(np.asarray(gt_win) == np.asarray(pred_win)))
        exact = 1.0 if pred_win == gt_win else 0.0

        # multiset overlap TP
        cp, cg = Counter(pred_win), Counter(gt_win)
        inter = set(cp.keys()) & set(cg.keys())
        tp = sum(min(cp[x], cg[x]) for x in inter)

        prec = tp / float(k)
        rec = tp / float(k)  # |GT| = k (Scheme A)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

        # MAP / NDCG for multiple relevant items
        gt_set = set(gt_win)

        hits = 0.0
        sum_prec = 0.0
        for i, p in enumerate(pred_win[:k]):
            if p in gt_set:
                hits += 1.0
                sum_prec += hits / (i + 1.0)
        denom = min(len(gt_set), k)
        ap = 0.0 if denom == 0 else sum_prec / denom

        dcg = 0.0
        for i, p in enumerate(pred_win[:k]):
            if p in gt_set:
                dcg += 1.0 / np.log2(i + 2)
        ideal_len = min(len(gt_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
        ndcg = 0.0 if idcg == 0 else dcg / idcg

        return {
            'pos_acc': pos_acc, 'exact': exact,
            'precision': prec, 'recall': rec, 'f1': f1,
            'map': ap, 'ndcg': ndcg
        }



    def compute_path_metric(self, y_prob_seq, y_true_seq, k_list=[3, 5, 7, 9]):
        """
        Path-level evaluation (Scheme A):
        - For each sequence, build windows of length k where future ground-truth length >= k.
        - For each window, compare the *predicted path* (greedy top-1 at each step) with the next k ground-truth items.

        Args:
            y_prob_seq: np.ndarray, shape (B, T, V)  (logits/probabilities per step)
            y_true_seq: np.ndarray, shape (B, T)     (ground-truth item ids per step)
            k_list: list[int]
        Returns:
            scores: dict[str, float] averaged over all valid windows (Scheme A skips short tails)
            scores_len: number of evaluated windows (for weighting across batches)
        Notes:
            - Tokens == PAD (0) or == 1 are treated as invalid and will truncate available horizon.
            - Metrics implemented:
                * acc@k: position-wise accuracy within the k-step path window
                * exact@k: 1 if the whole k-step path matches exactly, else 0
                * precision@k / recall@k / f1@k: based on multiset overlap between predicted items and GT items in the window
                * map@k / NDCG@k: treat the GT window as a relevant set (order-agnostic relevance)
        """
        y_prob_seq = np.asarray(y_prob_seq)
        y_true_seq = np.asarray(y_true_seq)

        B, T = y_true_seq.shape
        scores_len = 0

        # init containers
        scores = {f'acc@{k}': [] for k in k_list}
        scores.update({f'exact@{k}': [] for k in k_list})
        scores.update({f'precision@{k}': [] for k in k_list})
        scores.update({f'recall@{k}': [] for k in k_list})
        scores.update({f'f1@{k}': [] for k in k_list})
        scores.update({f'map@{k}': [] for k in k_list})
        scores.update({f'NDCG@{k}': [] for k in k_list})

        from collections import Counter

        def multiset_tp(pred_list, gt_list):
            cp, cg = Counter(pred_list), Counter(gt_list)
            inter = set(cp.keys()) & set(cg.keys())
            return sum(min(cp[x], cg[x]) for x in inter)

        def map_multi(gt_set, pred_list, k):
            # Average precision for multiple relevant items (gt_set), truncated at k
            hits = 0.0
            sum_prec = 0.0
            for i, p in enumerate(pred_list[:k]):
                if p in gt_set:
                    hits += 1.0
                    sum_prec += hits / (i + 1.0)
            denom = min(len(gt_set), k)
            return 0.0 if denom == 0 else sum_prec / denom

        def ndcg_multi(gt_set, pred_list, k):
            dcg = 0.0
            for i, p in enumerate(pred_list[:k]):
                if p in gt_set:
                    dcg += 1.0 / np.log2(i + 2)
            # ideal dcg: all relevant items ranked at top
            ideal_len = min(len(gt_set), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
            return 0.0 if idcg == 0 else dcg / idcg

        for b in range(B):
            # determine available horizon (Scheme A needs full k)
            gt_row = y_true_seq[b]
            # treat PAD(0) and token(1) as invalid/end
            valid_mask = (gt_row != self.PAD) & (gt_row != 1)
            valid_len = int(valid_mask.sum())
            if valid_len <= 0:
                continue

            # greedy path prediction at each step
            # y_prob_seq[b, t] -> vocab scores
            pred_row = y_prob_seq[b].argmax(axis=-1)

            for k in k_list:
                if valid_len < k:
                    # Scheme A: skip this sequence for this k (insufficient future)
                    continue

                # slide windows so we use as many supervised segments as possible
                for start in range(0, valid_len - k + 1):
                    gt_win = gt_row[start:start + k].tolist()
                    pred_win = pred_row[start:start + k].tolist()

                    scores_len += 1

                    # order-sensitive
                    pos_acc = float(np.mean(np.asarray(gt_win) == np.asarray(pred_win)))
                    exact = 1.0 if pred_win == gt_win else 0.0

                    # order-agnostic overlap (multiset)
                    tp = multiset_tp(pred_win, gt_win)
                    prec = tp / float(k)
                    rec = tp / float(k)  # |GT| = k
                    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

                    gt_set = set(gt_win)
                    ap = map_multi(gt_set, pred_win, k)
                    ndcg = ndcg_multi(gt_set, pred_win, k)

                    scores[f'acc@{k}'].append(pos_acc)
                    scores[f'exact@{k}'].append(exact)
                    scores[f'precision@{k}'].append(prec)
                    scores[f'recall@{k}'].append(rec)
                    scores[f'f1@{k}'].append(f1)
                    scores[f'map@{k}'].append(ap)
                    scores[f'NDCG@{k}'].append(ndcg)

        # average
        scores = {kk: (np.mean(vv) if len(vv) > 0 else 0.0) for kk, vv in scores.items()}
        return scores, scores_len

        def get_courses_by_video(self, video_name, course_video_mapping):
            """根据视频名称获取其所属的课程"""
            courses = []
            for course, videos in course_video_mapping.items():
                if video_name in videos:
                    courses.append(course)
            return courses


# Calculate accuracy of prediction result and its corresponding label
# output: tensor, labels: tensor
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)


class KTLoss(nn.Module):
    def __init__(self):
        super(KTLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')  # 不自动求平均

    def forward(self, pred_answers, real_answers, kt_mask):
        real_answers = real_answers[:, 1:].float()  # 截取有效部分并转为浮点
        answer_mask = kt_mask.bool()

        # # --- 计算 AUC 和 ACC ---
        # try:
        #     y_true = real_answers[answer_mask].cpu().detach().numpy()
        #     y_pred = pred_answers[answer_mask].cpu().detach().numpy()
        #     auc = roc_auc_score(y_true, y_pred)
        #     acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))  # 直接根据概率阈值计算ACC
        # except ValueError:
        #     auc, acc = -1, -1
        # --- 计算 AUC / ACC / Precision / Recall / F1 ---
        try:
            y_true = real_answers[answer_mask].cpu().detach().numpy()
            y_pred = pred_answers[answer_mask].cpu().detach().numpy()

            auc = roc_auc_score(y_true, y_pred)
            y_hat = (y_pred >= 0.5).astype(int)

            acc = accuracy_score(y_true, y_hat)
            precision = precision_score(y_true, y_hat, zero_division=0)
            recall = recall_score(y_true, y_hat, zero_division=0)
            f1 = f1_score(y_true, y_hat, zero_division=0)

        except ValueError:
            auc, acc, precision, recall, f1 = -1, -1, -1, -1, -1


        # --- 计算带掩码的 BCE 损失 ---
        loss_per_element = self.bce_loss(pred_answers, real_answers)
        valid_loss = loss_per_element * answer_mask.float()  # 应用掩码
        loss = valid_loss.sum() / answer_mask.float().sum()  # 仅对有效位置求平均

        # return loss, auc, acc
        return loss, auc, acc, precision, recall, f1



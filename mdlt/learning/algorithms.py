import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import copy
import numpy as np

import collections  


from mdlt.models import networks
from mdlt.utils.misc import count_samples_per_class, random_pairs_of_minibatches, ParamDict


ALGORITHMS = [
    'ERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'Fish',
    'ReSamp',
    'ReWeight',
    'SqrtReWeight',
    'CBLoss',
    'Focal',
    'LDAM',
    'BSoftmax',
    'CRT',
    'BoDA',
    'CAWRA_TAROT',
    'LODO_DA_MAML'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - return_feats()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, env_feats=None):
        """
        Perform one update step, given a list of (x, y) tuples for all envs.
        Admits an optional dict of features from each training domains.
        """
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, env_feats=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.network(x)


class ReSamp(ERM):
    """Naive resample, with no changes to ERM, but enable balanced sampling in hparams"""


class ReWeight(ERM):
    """Naive inverse re-weighting"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(ReWeight, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.weights_per_env = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            per_cls_weights = 1 / np.array(count_samples_per_class(labels, num_classes))
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
            self.weights_per_env[i] = torch.FloatTensor(per_cls_weights)

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x), y, weight=self.weights_per_env[env].to(device))
        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SqrtReWeight(ReWeight):
    """Square-root inverse re-weighting"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(SqrtReWeight, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.weights_per_env = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            per_cls_weights = 1 / np.sqrt(np.array(count_samples_per_class(labels, num_classes)))
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
            self.weights_per_env[i] = torch.FloatTensor(per_cls_weights)


class CBLoss(ReWeight):
    """Class-balanced loss, https://arxiv.org/pdf/1901.05555.pdf"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(CBLoss, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.weights_per_env = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            effective_num = 1. - np.power(self.hparams["beta"], count_samples_per_class(labels, num_classes))
            effective_num = np.array(effective_num)
            effective_num[effective_num == 1] = np.inf
            per_cls_weights = (1. - self.hparams["beta"]) / effective_num
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
            self.weights_per_env[i] = torch.FloatTensor(per_cls_weights)


class Focal(ERM):
    """Focal loss, https://arxiv.org/abs/1708.02002"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(Focal, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)

    @staticmethod
    def focal_loss(input_values, gamma):
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()

    def update(self, minibatches, env_feats=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = self.focal_loss(F.cross_entropy(self.predict(all_x), all_y, reduction='none'), self.hparams["gamma"])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class LDAM(ERM):
    """LDAM loss, https://arxiv.org/abs/1906.07413"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(LDAM, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.m_list = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            m_list = 1. / np.sqrt(np.sqrt(np.array(count_samples_per_class(labels, num_classes))))
            m_list = m_list * (self.hparams["max_m"] / np.max(m_list))
            self.m_list[i] = torch.FloatTensor(m_list)

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            x = self.predict(x)
            index = torch.zeros_like(x, dtype=torch.uint8)
            index.scatter_(1, y.data.view(-1, 1), 1)
            index_float = index.type(torch.FloatTensor)
            batch_m = torch.matmul(self.m_list[env][None, :].to(device), index_float.transpose(0, 1).to(device))
            batch_m = batch_m.view((-1, 1))
            x_m = x - batch_m
            output = torch.where(index, x_m, x)
            loss += F.cross_entropy(self.hparams["scale"] * output, y)
        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class BSoftmax(ERM):
    """Balanced softmax, https://arxiv.org/abs/2007.10740"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(BSoftmax, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.n_samples_per_env = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            n_samples_per_cls = np.array(count_samples_per_class(labels, num_classes))
            n_samples_per_cls[n_samples_per_cls == np.inf] = 1
            self.n_samples_per_env[i] = torch.FloatTensor(n_samples_per_cls)

    def update(self, minibatches, env_feats=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            x = self.predict(x)
            spc = self.n_samples_per_env[env].type_as(x)
            spc = spc.unsqueeze(0).expand(x.shape[0], -1)
            x = x + spc.log()
            loss += F.cross_entropy(input=x, target=y)
        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class CRT(ERM):
    """Classifier re-training with balanced sampling during the second earning stage"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(CRT, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )


class BoDA(ERM):
    """BoDA: balanced domain-class distribution alignment"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(BoDA, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.train_feats = None
        self.train_labels = None
        self.steps = 0
        self.nu = hparams["nu"]
        self.momentum = hparams["momentum"]
        self.temperature = hparams["temperature"]
        self.boda_start_step = hparams["boda_start_step"]
        self.feat_update_freq = hparams["feat_update_freq"]
        self.use_boda = hparams.get("use_boda", True)
        self.use_xent = bool(hparams.get("use_xent", True))
        self.use_calibration = bool(hparams.get("use_calibration", True))
        self.dist_measure = hparams.get("boda_dist_measure", "coral")

        # 'env_labels' can be None in evaluation, but not in training
        if env_labels is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_samples_table = torch.tensor([
                count_samples_per_class(env_labels[env], num_classes) for env in sorted(env_labels)]
            ).to(device)

            self.centroid_classes = torch.tensor(np.hstack([np.unique(env_labels[env]) for env in sorted(env_labels)])).to(device)
            self.centroid_envs = torch.tensor(np.hstack([
                i * np.ones_like(np.unique(env_labels[env])) for i, env in enumerate(sorted(env_labels))])).to(device)

            self.register_buffer('train_centroids', torch.zeros(self.centroid_classes.size(0), self.featurizer.n_outputs, device=device))

    @staticmethod
    def pairwise_dist(x, y):
        return torch.cdist(x, y)

    def macro_alignment_loss(self, x, y):
        # x 또는 y에 샘플이 2개 미만이면 penalty를 0으로 처리하여 계산 자체를 스킵
        if len(x) < 2 or len(y) < 2:
            return torch.tensor(0.0, device=x.device)
        
        # --boda_dist_measure 인자에 따라 거리 계산 방식 분기
        if self.dist_measure == 'mahalanobis':
            # ========================= [ 코드 수정 시작 ] =========================
            # 수치적 안정을 위해 double precision(float64)으로 계산 수행
            
            # 원래 dtype과 device 저장
            orig_dtype = x.dtype
            
            # float64로 변환
            x = x.double()
            y = y.double()

            # --- Mahalanobis Distance (in float64) ---
            mean_x = x.mean(0)
            mean_y = y.mean(0)
            mean_diff = mean_x - mean_y

            # Pooled covariance matrix
            n_x, n_y = len(x), len(y)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cov_x = (cent_x.t() @ cent_x) / (n_x - 1)
            cov_y = (cent_y.t() @ cent_y) / (n_y - 1)
            pooled_cov = ((n_x - 1) * cov_x + (n_y - 1) * cov_y) / (n_x + n_y - 2)
            
            # 수치적 안정을 위한 정규화 (Regularization) - 값을 약간 올림
            d = pooled_cov.shape[0]
            reg = 1e-4  # 기존 1e-5에서 상향 조정
            try:
                # 역행렬 계산
                inv_pooled_cov = torch.linalg.inv(pooled_cov + torch.eye(d, device=x.device, dtype=torch.float64) * reg)
            except torch.linalg.LinAlgError:
                # 역행렬 계산이 불가능한 경우 패널티를 0으로 처리
                return torch.tensor(0.0, device=x.device)

            # Mahalanobis distance squared: (diff.T) @ (inv_cov) @ (diff)
            md_sq = mean_diff.view(1, -1) @ inv_pooled_cov @ mean_diff.view(-1, 1)
            
            # 최종 penalty 값은 원래 dtype으로 변환
            penalty = md_sq.squeeze().to(orig_dtype)

        else: # Default: 'coral'
            # --- Original CORAL-like Distance ---
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            mean_diff = (mean_x - mean_y).pow(2).mean()

            cent_x = x - mean_x
            cent_y = y - mean_y
            
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)
            
            cova_diff = (cova_x - cova_y).pow(2).mean()
            penalty = mean_diff + cova_diff

        # 계산 결과가 NaN/inf인지 한 번 더 확인
        if torch.isnan(penalty) or torch.isinf(penalty):
            return torch.tensor(0.0, device=x.device)

        return penalty
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★


    def update_feature_stats(self, env_feats):
        if self.steps == 0 or self.steps % self.feat_update_freq != 0:
            return

        train_feats = [torch.stack(x, dim=0) for x in env_feats['feats'].values()]
        train_labels = [torch.stack(x, dim=0) for x in env_feats['labels'].values()]

        curr_centroids = torch.empty((0, self.train_centroids.size(-1))).to(train_feats[0].device)
        for env in range(len(train_feats)):
            curr_centroids = torch.cat((
                curr_centroids,
                torch.stack([train_feats[env][torch.where(train_labels[env] == c)[0]].mean(0)
                             for c in torch.unique(train_labels[env])])
            ))
        factor = 0 if self.steps == self.feat_update_freq else self.momentum
        self.train_centroids = \
            (1 - factor) * curr_centroids.to(self.train_centroids.device) + factor * self.train_centroids

    def update(self, minibatches, env_feats=None):
        # ─── feature stats 업데이트 ─────────────────────────────────────────
        self.update_feature_stats(env_feats)

        device = next(self.parameters()).device
        n_envs = len(minibatches)

        # 전체 y, env 지시자(cursor) 벡터
        all_y    = torch.cat([y for _, y in minibatches])
        all_envs = torch.cat([env * torch.ones_like(y)
                              for env, (_, y) in enumerate(minibatches)])

        # ─── 디버깅 1: 모델 Forward Pass 결과 확인 ────────────────
        features    = [self.featurizer(x) for x, _ in minibatches]
        assert not any(torch.isnan(f).any() for f in features), f"NaN detected in features at step {self.steps}"
        
        classifiers = [self.classifier(f)   for f in features]
        classifiers = [torch.clamp(c, min=-100, max=100) for c in classifiers]
        assert not any(torch.isnan(c).any() for c in classifiers), f"NaN detected in classifiers (logits) at step {self.steps}"

        # ① Cross‐entropy loss
        if self.use_xent:
            loss_x = sum(F.cross_entropy(classifiers[i], minibatches[i][1]) for i in range(n_envs)) / n_envs
        else:
            loss_x = torch.tensor(0.0, device=device)
        assert not torch.isnan(loss_x), f"NaN detected in loss_x at step {self.steps}"

        # ② BoDA loss (warm-up 이후 & use_boda=True)
        if self.steps >= self.boda_start_step and self.use_boda:
            # 1) pairwise distance
            pdist = -self.pairwise_dist(self.train_centroids,
                                        torch.cat(features))

            # 2) balanced distance (분모 clamp)
            denom_per = self.n_samples_table[all_envs.long(), all_y.long()]\
                            .clamp_min(1).to(device)
            logits = pdist / denom_per

            # 3) calibration weights (분모 clamp)
            n_num = self.n_samples_table[self.centroid_envs.long(),
                                         self.centroid_classes.long()]\
                        .clamp_min(1)
            n_den = self.n_samples_table[all_envs.long(), all_y.long()]\
                        .clamp_min(1)
            H, W = n_num.size(0), n_den.size(0)
            base_w = (n_num.unsqueeze(1).expand(-1, W) /
                      n_den.unsqueeze(0).expand(H, -1)) ** self.nu

            same_env = (self.centroid_envs.view(-1, 1) ==
                        all_envs.view(1, -1))
            gamma    = self.hparams.get("cross_env_gamma", 1.0)
            cross_w  = torch.where(same_env, 1.0, gamma)
            cal_w    = base_w * cross_w

            if self.use_calibration:
                logits = logits * cal_w.to(device)

            # 4) temperature scaling
            logits = logits / self.temperature

            # 5) log-prob 계산
            mask_same = (self.centroid_classes.view(-1, 1) == 
                         all_y.view(1, -1)) & same_env
            log_prob = logits - torch.log(
                (torch.exp(logits) * (~mask_same)).sum(0, keepdim=True)
                + 1e-12
            )

            # 6) positive-pair mask
            mask_cls = (self.centroid_classes.view(-1, 1) == 
                        all_y.view(1, -1))
            pos_mask = mask_cls & (~same_env)
            lp_pos   = log_prob * pos_mask.float()

            # 7) denom 방어
            cnt_pos = pos_mask.sum()
            if cnt_pos > 0:
                loss_b = - lp_pos.sum() / cnt_pos
            else:
                loss_b = torch.tensor(0.0, device=device)
        else:
            loss_b = torch.tensor(0.0, device=device)
        assert not torch.isnan(loss_b), f"NaN detected in loss_b at step {self.steps}"


        # ③ Macro alignment penalty (warm-up 이후에만)
        if self.steps >= self.boda_start_step:
            penalty = 0.0
            for i in range(n_envs):
                for j in range(i+1, n_envs):
                    # ========================= [ 코드 수정 시작 ] =========================
                    # penalty 값에 log1p를 적용하여 스케일을 안정화시킵니다.
                    penalty += torch.log1p(self.macro_alignment_loss(
                        features[i], features[j]))
                    # ========================= [  코드 수정 끝  ] =========================
            if n_envs > 1:
                penalty = penalty / (n_envs*(n_envs-1)/2)

        else:
            penalty = torch.tensor(0.0, device=device)
        assert not torch.isnan(penalty), f"NaN detected in penalty at step {self.steps}"


        # ④ Global imbalance loss: global_weight > 0 인 경우에만 계산
        device = next(self.parameters()).device
        gw = self.hparams.get('global_weight', 0.0)
        if gw > 0:
            eps = 1e-8
            
            # 1. Calculate target distribution q safely
            global_counts = self.n_samples_table.sum(dim=0).float()
            total_samples = global_counts.sum()
            
            if total_samples > 0:
                q = global_counts / total_samples
            else:
                # If no samples, use a uniform distribution as a fallback
                num_classes = self.classifier.out_features
                q = torch.full((num_classes,), 1.0 / num_classes, device=device)

            # 2. Calculate predicted distribution p_bar safely
            all_logits = torch.cat(classifiers, dim=0)
            # 수정 코드: log_softmax를 통해 수치적으로 안정하게 softmax 계산
            log_probs = F.log_softmax(all_logits, dim=1)
            probs = torch.exp(log_probs)
            p_bar = probs.mean(dim=0)

            # 3. Calculate KL divergence robustly
            # --- 안전한 KL 계산 ------------------------------------
            p_bar = p_bar.clamp_min(eps)
            # 0 확률은 그대로 두고 음수만 방어
            q = q.clamp_min(0)

            mask = q > 0
            if mask.any():
                p_sel = p_bar[mask]
                q_sel = q[mask]

                p_sel = p_sel / p_sel.sum()
                q_sel = q_sel / q_sel.sum()

                global_loss = torch.sum(q_sel * (q_sel.log() - p_sel.log()))
            else:
                global_loss = torch.tensor(0.0, device=device)

        else:
            # global_weight == 0이면 loss에 0 곱해지도록 0으로 처리
            global_loss = torch.tensor(0.0, device=device)
        assert not torch.isnan(global_loss), f"NaN detected in global_loss at step {self.steps}"

        # ⑤ 최종 loss 조합
        macro_w = self.hparams['macro_weight'] if self.steps >= self.boda_start_step else 0.0
        loss = (
            loss_x
            + macro_w * penalty
            + self.hparams['boda_weight'] * loss_b
            + self.hparams.get('global_weight', 0.0) * global_loss
        )
        assert not torch.isnan(loss), f"NaN detected in final loss at step {self.steps}"


        # (디버깅용 NaN 체크 — 필요시 활성화)
        for name, v in [('loss_x',loss_x),('loss_b',loss_b),
                       ('penalty',penalty),('global',global_loss)]:
            if torch.isnan(v):
                print(f"NaN in {name} at step {self.steps}")

        # ─── backward & step ───────────────────────────────────────────────
        torch.autograd.set_detect_anomaly(True)  # anomaly detection
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)


        self.optimizer.step()
        self.steps += 1

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), f"Loss explosion: {loss.item()}"

        # ─── 반환값 ────────────────────────────────────────────────────────
        out = {'loss': loss_x.item(), 'penalty': float(penalty)}
        if self.steps > self.boda_start_step and self.use_boda:
            out['boda_loss'] = loss_b.item()
        return out

class CAWRA_TAROT(BoDA):
    """
    CAWRA-TAROT: Class-distribution-Aware Weighted Robust Adaptation
    BoDA를 기반으로 TAROT의 적대적 강건성과 CAWRA의 클래스 가중치 보정 아이디어를 통합합니다.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        # [CAWRA-TAROT] 부모 클래스(BoDA)의 __init__을 먼저 호출합니다.
        super(CAWRA_TAROT, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)

        # [CAWRA-TAROT] 적대적 학습(Adversarial Training) 파라미터를 추가합니다.
        self.pgd_eps = hparams.get("pgd_eps", 8.0 / 255.0)
        self.pgd_alpha = hparams.get("pgd_alpha", 2.0 / 255.0)
        self.pgd_steps = hparams.get("pgd_steps", 10)

        # [CAWRA-TAROT] CAWRA 가중치를 계산합니다.
        if env_labels is not None:
            # 가중치 beta_{d,c} = (c 클래스의 다른 도메인 평균 샘플 수) / (d 도메인의 c 클래스 샘플 수)
            counts = self.n_samples_table.float().clamp_min(1.0) # 0으로 나누는 것을 방지
            total_counts_per_class = counts.sum(dim=0)
            
            # (전체 합 - 자기 자신) / (도메인 수 - 1)
            avg_counts_other_domains = (total_counts_per_class.unsqueeze(0) - counts) / (num_domains - 1)
            # 0으로 나누는 것을 방지하기 위해 분모가 0일 경우 분자도 0으로 만들어 0/0 -> nan 대신 0이 되게 함
            avg_counts_other_domains[counts == 0] = 0

            self.cawra_weights = avg_counts_other_domains / counts
            self.cawra_weights[torch.isnan(self.cawra_weights)] = 1.0 # 0/0 -> nan을 1로 처리
            self.cawra_weights = self.cawra_weights.clamp_max(hparams.get("cawra_clip", 10.0))
            
            print("--- CAWRA Weights Initialized ---")
            print(self.cawra_weights)
            print("---------------------------------")


    def pgd_attack(self, images, labels):
        """[CAWRA-TAROT] PGD 공격을 수행하여 적대적 예제를 생성하는 함수"""
        images_adv = images.clone().detach().requires_grad_(True)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.pgd_steps):
            outputs = self.network(images_adv)
            self.network.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()

            attack_grad = images_adv.grad.sign()
            images_adv = images_adv.detach() + self.pgd_alpha * attack_grad
            total_noise = torch.clamp(images_adv - images, -self.pgd_eps, self.pgd_eps)
            images_adv = torch.clamp(images + total_noise, 0, 1).detach().requires_grad_(True)
            
        return images_adv.detach()

    def forward(self, x):
        """
        모델 객체가 함수처럼 호출될 때 실행되는 정방향 연산을 정의합니다.
        'self.network'는 ERM 클래스에서 정의된 전체 모델 (featurizer + classifier)입니다.
        """
        return self.network(x)

    def update(self, minibatches, env_feats=None):
        device = minibatches[0][0].device
        
        # ─── 1. 적대적 예제 생성 (TAROT 파트) ────────────────────────────────
        all_x = torch.cat([x for x, y in minibatches])
        all_y_orig = torch.cat([y for x, y in minibatches])
        
        with torch.no_grad():
            pseudo_logits = self(all_x)
            pseudo_labels = pseudo_logits.argmax(dim=1)
        
        all_x_adv = self.pgd_attack(all_x, pseudo_labels)

        # 도메인별로 적대적 예제를 다시 분리
        adv_minibatches = []
        start_idx = 0
        for x, y in minibatches:
            end_idx = start_idx + len(x)
            adv_minibatches.append((all_x_adv[start_idx:end_idx], y))
            start_idx = end_idx

        # ─── 2. 피쳐 추출 및 손실 계산 (BoDA, CAWRA 파트) ──────────────────
        if env_feats:
            self.update_feature_stats(env_feats)
        
        n_envs = len(minibatches)
        all_envs = torch.cat([env * torch.ones_like(y) for env, (_, y) in enumerate(minibatches)])

        # '적대적 예제'로부터 피쳐와 로짓을 계산
        features = [self.featurizer(x_adv) for x_adv, _ in adv_minibatches]
        classifiers = [self.classifier(f) for f in features]
        all_logits = torch.cat(classifiers)

        # ① [CAWRA-TAROT] 가중치가 적용된 Cross‐entropy loss
        if self.use_xent:
            per_sample_loss_x = F.cross_entropy(all_logits, all_y_orig, reduction='none')
            weights = self.cawra_weights[all_envs.long(), all_y_orig.long()].to(device)
            loss_x = (per_sample_loss_x * weights).mean()
        else:
            loss_x = torch.tensor(0.0, device=device)

        # ② BoDA loss (입력 피쳐가 적대적 피쳐임)
        loss_b = torch.tensor(0.0, device=device)
        if self.steps >= self.boda_start_step and self.use_boda:
            pdist = -self.pairwise_dist(self.train_centroids, torch.cat(features))
            denom_per = self.n_samples_table[all_envs.long(), all_y_orig.long()].clamp_min(1).to(device)
            logits = pdist / denom_per
            
            n_num = self.n_samples_table[self.centroid_envs.long(), self.centroid_classes.long()].clamp_min(1)
            n_den = self.n_samples_table[all_envs.long(), all_y_orig.long()].clamp_min(1)
            H, W = n_num.size(0), n_den.size(0)
            base_w = (n_num.unsqueeze(1).expand(-1, W) / n_den.unsqueeze(0).expand(H, -1)) ** self.nu
            same_env = (self.centroid_envs.view(-1, 1) == all_envs.view(1, -1))
            gamma = self.hparams.get("cross_env_gamma", 1.0)
            cross_w = torch.where(same_env, 1.0, gamma)
            cal_w = base_w * cross_w
            if self.use_calibration:
                logits = logits * cal_w.to(device)

            logits = logits / self.temperature
            mask_same = (self.centroid_classes.view(-1, 1) == all_y_orig.view(1, -1)) & same_env
            log_prob = logits - torch.log((torch.exp(logits) * (~mask_same)).sum(0, keepdim=True) + 1e-12)
            mask_cls = (self.centroid_classes.view(-1, 1) == all_y_orig.view(1, -1))
            pos_mask = mask_cls & (~same_env)
            lp_pos = log_prob * pos_mask.float()
            
            cnt_pos = pos_mask.sum()
            if cnt_pos > 0:
                loss_b = -lp_pos.sum() / cnt_pos

        # ③ Macro alignment penalty (입력 피쳐가 적대적 피쳐임)
        penalty = torch.tensor(0.0, device=device)
        if self.steps >= self.boda_start_step:
            current_penalty = 0.0
            for i in range(n_envs):
                for j in range(i + 1, n_envs):
                    current_penalty += torch.log1p(self.macro_alignment_loss(features[i], features[j]))
            if n_envs > 1:
                penalty = current_penalty / (n_envs * (n_envs - 1) / 2)
        
        # ④ [CAWRA-TAROT] 타겟 도메인 강건성 손실 추가
        target_adv_loss = F.cross_entropy(all_logits, pseudo_labels)

        # ⑤ 최종 loss 조합
        macro_w = self.hparams['macro_weight'] if self.steps >= self.boda_start_step else 0.0
        boda_w = self.hparams['boda_weight']
        target_adv_w = self.hparams.get('target_adv_weight', 1.0) # 새로운 하이퍼파라미터

        loss = (
            loss_x  # CAWRA 가중치 적용됨
            + macro_w * penalty
            + boda_w * loss_b
            + target_adv_w * target_adv_loss # TAROT 강건성 항 추가
        )

        # ─── backward & step ───────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1

        return {
            'loss': loss.item(), 
            'loss_x_weighted': loss_x.item(), 
            'loss_boda': loss_b.item(), 
            'penalty_macro': penalty.item(), 
            'loss_target_adv': target_adv_loss.item()
        }

class Fish(Algorithm):
    """Gradient Matching for Domain Generalization, Shi et al. 2021."""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(Fish, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                                weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    @staticmethod
    def fish(meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, env_feats=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):
        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g:
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams['lambda'] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(DANN, self).__init__(
            input_shape, num_classes, num_domains, hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(CDANN, self).__init__(
            input_shape, num_classes, num_domains, hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)

    def update(self, minibatches, env_feats=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()
        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)

    def update(self, minibatches, env_feats=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)
        self.optimizer.step()

        return {'loss': objective}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    @staticmethod
    def my_cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, env_feats=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifiers = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifiers[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma'] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning (https://arxiv.org/abs/1711.07910)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, env_feats=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + \
                               (1 - self.ema) * self.embeddings[env]
            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"], weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    @staticmethod
    def randomize(x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, env_feats=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(), 'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))

class LODO_DA_MAML(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(LODO_DA_MAML, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams.get('nonlinear_classifier', False))
        
        self.all_params = list(self.featurizer.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(
            self.all_params,
            lr=self.hparams["meta_lr"],
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )
        
        if env_labels:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_samples_table = torch.tensor([
                count_samples_per_class(env_labels[env], num_classes) for env in sorted(env_labels.keys())]
            ).to(device)
        else:
            self.n_samples_table = torch.ones(num_domains, num_classes)

    @staticmethod
    def _pairwise_l2_distance(x, y):
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        inner_prod = torch.mm(x, y.t())
        dist_sq = x_norm - 2 * inner_prod + y_norm.t()
        return torch.sqrt(dist_sq.clamp_min(1e-12))

    def _compute_centroids(self, source_minibatches_with_indices):
        all_features, all_labels, all_domains = [], [], []
        with torch.no_grad():
            for domain_idx, (x, y) in source_minibatches_with_indices:
                features = self.featurizer(x)
                features = F.normalize(features, p=2, dim=1)
                all_features.append(features)
                all_labels.append(y)
                all_domains.append(torch.full_like(y, domain_idx))
        
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        all_domains = torch.cat(all_domains)
        
        unique_domain_classes = torch.unique(torch.stack([all_domains, all_labels], dim=1), dim=0)
        centroids = torch.zeros(len(unique_domain_classes), all_features.shape[1], device=all_features.device)
        
        for i, (d, c) in enumerate(unique_domain_classes):
            mask = (all_domains == d) & (all_labels == c)
            if mask.sum() > 0:
                centroids[i] = all_features[mask].mean(dim=0)
        return centroids, unique_domain_classes

    def _compute_da_loss(self, features, labels, domains, centroids, centroid_domain_classes):
        device = features.device
        if centroids.shape[0] == 0:
            return torch.tensor(0.0, device=device)
        
        normalized_features = F.normalize(features, p=2, dim=1)
        logits = -self._pairwise_l2_distance(centroids, normalized_features)
        
        centroid_domains = centroid_domain_classes[:, 0].view(-1, 1)
        sample_domains = domains.view(1, -1)
        same_env_mask = (centroid_domains == sample_domains)
        
        logits = logits / self.hparams.get('temperature', 1.0)
        centroid_labels = centroid_domain_classes[:, 1].view(-1, 1)
        sample_labels = labels.view(1, -1)
        mask_cls = (centroid_labels == sample_labels)
        pos_mask = mask_cls & (~same_env_mask)
        cnt_pos = pos_mask.sum(dim=0) 
        valid_samples_mask = cnt_pos > 0
        
        if not valid_samples_mask.any():
            return torch.tensor(0.0, device=device)
        
        logits_valid = logits[:, valid_samples_mask]
        pos_mask_valid = pos_mask[:, valid_samples_mask]
        
        # Negative masking
        neg_logits = logits_valid.clone()
        neg_logits[pos_mask_valid] = -float('inf')
        
        log_sum_exp = torch.logsumexp(neg_logits, dim=0, keepdim=True)
        log_prob = logits_valid - log_sum_exp
        lp_pos = log_prob[pos_mask_valid]
        
        if lp_pos.numel() == 0:
            return torch.tensor(0.0, device=device)
            
        loss = -lp_pos.mean()
        return loss

    def meta_update(self, minibatches, held_out_domain_idx):
        # 소스 도메인과 타겟 도메인 분리
        source_minibatches_with_indices = [(i, mb) for i, mb in enumerate(minibatches) if i != held_out_domain_idx]
        if not source_minibatches_with_indices:
            return {'meta_loss_ce': 0.0, 'meta_loss_da': 0.0, 'total_loss': 0.0}

        # 1. 소스 도메인들의 중심점(Centroid) 미리 계산
        source_centroids, source_domain_classes = self._compute_centroids(source_minibatches_with_indices)

        # 2. Inner Loop: 소스 도메인들로 CE+DA Loss를 이용한 가상 학습
        featurizer_fast_weights = collections.OrderedDict(self.featurizer.named_parameters())
        classifier_fast_weights = collections.OrderedDict(self.classifier.named_parameters())
        
        inner_losses = []
        for i, (x, y) in source_minibatches_with_indices:
            features = self.featurizer.forward_with_params(x, featurizer_fast_weights)
            logits = self.classifier.forward_with_params(features, classifier_fast_weights)
            
            loss_ce = F.cross_entropy(logits, y)
            d = torch.full_like(y, i)
            loss_da = self._compute_da_loss(features, y, d, source_centroids, source_domain_classes)
            
            combined_loss = loss_ce + self.hparams['lambda_da'] * loss_da
            inner_losses.append(combined_loss)

        inner_loss = torch.mean(torch.stack(inner_losses))
        
        all_fast_weights = list(featurizer_fast_weights.values()) + list(classifier_fast_weights.values())
        grads = torch.autograd.grad(inner_loss, all_fast_weights, create_graph=False)
        
        num_featurizer_params = len(featurizer_fast_weights)
        adapted_featurizer_weights = collections.OrderedDict()
        adapted_classifier_weights = collections.OrderedDict()
        for i, (name, param) in enumerate(self.featurizer.named_parameters()):
            adapted_featurizer_weights[name] = param - self.hparams['inner_lr'] * grads[i]
        for i, (name, param) in enumerate(self.classifier.named_parameters()):
            adapted_classifier_weights[name] = param - self.hparams['inner_lr'] * grads[num_featurizer_params + i]
            
        # 3. Outer Loop: 타겟 도메인으로 CE+DA Loss를 이용한 평가
        q_x, q_y = minibatches[held_out_domain_idx]

        q_features = self.featurizer.forward_with_params(q_x, adapted_featurizer_weights)
        q_logits = self.classifier.forward_with_params(q_features, adapted_classifier_weights)
        
        meta_loss_ce = F.cross_entropy(q_logits, q_y)

        meta_loss = meta_loss_ce
                
        # 4. Meta-Update: 원본 모델 업데이트
        total_outer_loss = meta_loss

        self.optimizer.zero_grad()
        total_outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=1.0) # 이 줄 추가
        self.optimizer.step()
        
        return {
            'meta_loss_ce': meta_loss_ce.item(),
            'meta_loss_da': meta_loss_da.item(),
            'total_loss': total_outer_loss.item()
        }

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def return_feats(self, x):
        return self.featurizer(x)

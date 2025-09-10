import argparse
import collections
import json
import os
import random
import sys
import time
import shutil
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from tensorboard_logger import Logger

from mdlt import hparams_registry
from mdlt.dataset import datasets
from mdlt.learning import algorithms
from mdlt.utils import misc
from mdlt.dataset.fast_dataloader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import TensorDataset


from tensorboard_logger import Logger
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# SubsetWithTargets: torch.utils.data.Subset 에 .targets 속성까지 복사
# ──────────────────────────────────────────────────────────────────────────────
class SubsetWithTargets(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        if hasattr(dataset, "targets"):            # TensorDataset, SplitImageFolder 등
            self.targets = np.asarray(dataset.targets)[indices]
        elif hasattr(dataset, "tensors"):          # TensorDataset
            self.targets = dataset.tensors[1].numpy()[indices]
        else:
            raise AttributeError(
                "custom_counts: 대상 데이터셋에 targets 속성을 찾을 수 없습니다."
            )

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Domain LT')
    parser.add_argument('--algorithm', type=str, default="ERM", 
                        choices=algorithms.ALGORITHMS)
    
    parser.add_argument('--use_meta_learning', type=str2bool, nargs='?', const=True, default=False,
                        help='If true, use the MAML-style meta-learning loop.')    
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='Meta learning rate (eta)')
    parser.add_argument('--inner_lr', type=float, default=1e-3, help='Inner loop adaptation learning rate (alpha)')
    parser.add_argument('--meta_beta', type=float, default=1.0, help='Weight for the meta-loss')
    parser.add_argument('--lambda_da', type=float, default=0.1, help='Weight for the DA loss')

    parser.add_argument('--m_way_range', type=int, nargs=2, default=[2, 5], help='Range of m for m-way classification (min, max)')
    parser.add_argument('--k_shot_range', type=int, nargs=2, default=[1, 5], help='Range of K for K-shot (support set)')
    parser.add_argument('--k_query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--r_imb_range', type=float, nargs=2, default=[1.0, 10.0], help='Range of imbalance ratio r (min, max)')
    
    parser.add_argument('--dataset', type=str, default="PACS", choices=datasets.DATASETS)
    parser.add_argument('--output_folder_name', type=str, default='debug')
    parser.add_argument('--imb_type', type=str, default="eeee",
                        help='Length should be equal to # of envs, each refers to imb_type within that env')
    parser.add_argument('--imb_factor', type=float, default=0.1)

    parser.add_argument('--custom_counts', type=str, default=None,
                        help='JSON list (len=#env) of per‑class counts for *train* split')
    parser.add_argument('--cross_env_gamma',
                        type=float,
                        default=None,
                        help='(BoDA 전용) cross_env_gamma 값을 수동으로 override')
    parser.add_argument(
        '--use_boda',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to apply BoDA loss (True/False).'
    )
    parser.add_argument(
        '--use_xent',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to include cross-entropy loss (True/False)'
    )
    parser.add_argument(
        '--use_calibration',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to include calibration (True/False)'
    )
    parser.add_argument(
        '--boda_dist_measure',
        type=str,
        default='coral',
        choices=['coral', 'mahalanobis'],
        help='(BoDA only) Distance measure for macro-alignment penalty.'
    )
    parser.add_argument('--global_weight',
                        type=float,
                        default=0.0,
                        help='Weight β for the global imbalance loss term')
    parser.add_argument('--macro_weight',
                        type=float,
                        default=None,
                        help='(BoDA only) Weight for the macro-alignment penalty.')
    parser.add_argument('--target_adv_weight', type=float, default=None,
                        help='(CAWRA_TAROT only) Weight for the target adversarial loss.')
    parser.add_argument('--pgd_eps', type=float, default=None,
                        help='(CAWRA_TAROT only) Epsilon for PGD attack.')
    parser.add_argument('--data_dir', type=str, default="/home/shared")

    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 for "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--selected_envs', type=int, nargs='+', default=None, help='Train only on selected envs')
    parser.add_argument('--stage1_folder', type=str, default='vanilla')
    parser.add_argument('--stage1_algo', type=str, default='ERM')
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    args = parser.parse_args()

    start_step = 0
    args.best_val_acc = 0
    best_env_acc = {}
    best_shot_acc = {}
    best_class_acc = collections.defaultdict(list)
    store_prefix = f"{args.dataset}_{args.imb_type}_{args.imb_factor}" if 'Imbalance' in args.dataset else args.dataset
    args.store_name = f"{store_prefix}_{args.algorithm}_hparams{args.hparams_seed}_seed{args.seed}"
    if args.selected_envs is not None:
        args.store_name = f"{args.store_name}_env{str(args.selected_envs).replace(' ', '')[1:-1]}"

    misc.prepare_folders(args)
    args.output_dir = os.path.join(args.output_dir, args.output_folder_name, args.store_name)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    tb_logger = Logger(logdir=args.output_dir, flush_secs=2)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if 'Imbalance' in args.dataset:
        hparams.update({'imb_type_per_env': [misc.IMBALANCE_TYPE[x] for x in args.imb_type],
                        'imb_factor': args.imb_factor})
    if 'BoDA' in args.algorithm or args.algorithm == 'CAWRA_TAROT':
        if args.cross_env_gamma is not None:
            hparams['cross_env_gamma'] = args.cross_env_gamma
        hparams['use_boda'] = args.use_boda
        hparams['use_xent'] = args.use_xent
        hparams['global_weight'] = args.global_weight
        hparams['use_calibration'] = args.use_calibration
        hparams['boda_dist_measure'] = args.boda_dist_measure
        if args.macro_weight is not None:
            hparams['macro_weight'] = args.macro_weight

    if args.algorithm == 'CAWRA_TAROT':
        if args.target_adv_weight is not None:
            hparams['target_adv_weight'] = args.target_adv_weight
        if args.pgd_eps is not None:
            hparams['pgd_eps'] = args.pgd_eps / 255.0 

    if args.custom_counts:
        hparams['custom_counts'] = json.loads(args.custom_counts)
    else:
        hparams['custom_counts'] = None

    hparams.update({
        'meta_lr': args.meta_lr,
        'inner_lr': args.inner_lr,
        'meta_beta': args.meta_beta,
        'lambda_da': args.lambda_da
    })

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset in vars(datasets):
        train_dataset = vars(datasets)[args.dataset](args.data_dir, 'train', hparams)
        val_dataset = vars(datasets)[args.dataset](args.data_dir, 'val', hparams)
        test_dataset = vars(datasets)[args.dataset](args.data_dir, 'test', hparams)
        test_dataset = vars(datasets)[args.dataset](args.data_dir, 'test', hparams)

        if hparams.get('custom_counts'):
            c_counts = hparams['custom_counts']
            assert len(c_counts) == len(train_dataset), \
                f"custom_counts: 도메인 수가 {len(train_dataset)}인데 {len(c_counts)}개가 전달됨"

            for env_i in range(len(train_dataset)):
                env_ds = train_dataset[env_i]         
                keep_per_cls = c_counts[env_i]
                num_classes = len(keep_per_cls)

                # 라벨 벡터 획득
                if hasattr(env_ds, "targets"):
                    labels_np = np.asarray(env_ds.targets).astype(int)
                elif hasattr(env_ds, "tensors"):
                    labels_np = env_ds.tensors[1].numpy().astype(int)
                else:
                    raise AttributeError("Dataset에 targets 배열이 없습니다.")

                assert num_classes == labels_np.max() + 1, \
                    "custom_counts 내부 리스트 길이(클래스 수)가 실제 클래스 수와 다릅니다."

                sel_idx = []
                for cls, n_keep in enumerate(keep_per_cls):
                    cls_idx = np.where(labels_np == cls)[0]
                    assert n_keep <= len(cls_idx), \
                        f"env{env_i}-class{cls}: 요청 {n_keep} > 보유 {len(cls_idx)}"
                    np.random.shuffle(cls_idx)
                    sel_idx.extend(cls_idx[:n_keep])

                train_dataset.datasets[env_i] = SubsetWithTargets(env_ds, sel_idx)

    else:
        raise NotImplementedError

    def sample_task_parameters(args):
        m = np.random.randint(args.m_way_range[0], args.m_way_range[1] + 1)
        k_s = np.random.randint(args.k_shot_range[0], args.k_shot_range[1] + 1)
        r = np.random.uniform(args.r_imb_range[0], args.r_imb_range[1])
        return m, k_s, r

    def create_task_set(dataset, max_samples_per_task=100, support_ratio=0.5):
        num_samples = len(dataset)
        if num_samples < 2:
            return None, None

        if num_samples > max_samples_per_task:
            indices = np.random.choice(num_samples, max_samples_per_task, replace=False)
        else:
            indices = np.arange(num_samples)
        
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * support_ratio)
        
        if split_idx == 0:
            split_idx = 1
        elif split_idx == len(indices):
            split_idx = len(indices) - 1

        support_indices = indices[:split_idx]
        query_indices = indices[split_idx:]

        if len(support_indices) == 0 or len(query_indices) == 0:
            return None, None

        support_samples = [dataset[i] for i in support_indices]
        query_samples = [dataset[i] for i in query_indices]
        
        support_x_list, support_y_list = zip(*support_samples)
        query_x_list, query_y_list = zip(*query_samples)
        
        support_x = torch.stack(support_x_list)
        support_y = torch.tensor(support_y_list, dtype=torch.long)
        
        query_x = torch.stack(query_x_list)
        query_y = torch.tensor(query_y_list, dtype=torch.long)
        
        return (support_x, support_y), (query_x, query_y)

    num_workers = train_dataset.N_WORKERS
    input_shape = train_dataset.input_shape
    num_classes = train_dataset.num_classes
    n_steps = args.steps or train_dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ
    many_shot_thr = train_dataset.MANY_SHOT_THRES
    few_shot_thr = train_dataset.FEW_SHOT_THRES

    if args.selected_envs is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, args.selected_envs)
        val_dataset = torch.utils.data.Subset(val_dataset, args.selected_envs)
        test_dataset = torch.utils.data.Subset(test_dataset, args.selected_envs)
    env_ids = args.selected_envs if args.selected_envs is not None else np.arange(len(train_dataset))

    print("Dataset:")
    from mdlt.utils.misc import kl_divergence
    train_cls_cnt, test_cls_cnt = {}, {}

    header = ['env'] + [f'c{c}' for c in range(num_classes)] + ['total']
    misc.print_row(header, colwidth=8)

    for i, (tr, _, te) in enumerate(zip(train_dataset, val_dataset, test_dataset)):
        t_tr = tr.targets if 'Imbalance' not in args.dataset else tr.tensors[1].numpy()
        t_te = te.targets if 'Imbalance' not in args.dataset else te.tensors[1].numpy()
        cnt_tr = np.bincount(t_tr, minlength=num_classes)
        cnt_te = np.bincount(t_te, minlength=num_classes)
        train_cls_cnt[f'env{env_ids[i]}'] = cnt_tr
        test_cls_cnt[f'env{env_ids[i]}']  = cnt_te

        misc.print_row([f'env{env_ids[i]}'] + cnt_tr.tolist() + [cnt_tr.sum()], colwidth=8)

    print("\n[KL(train env_i ‖ train env_j)]")
    for i in range(len(env_ids)):
        for j in range(i + 1, len(env_ids)):
            ei, ej = f'env{env_ids[i]}', f'env{env_ids[j]}'
            kl = kl_divergence(train_cls_cnt[ei], train_cls_cnt[ej])
            print(f'{ei} ‖ {ej}: {kl:.4f}')

    print("\n[KL(train ‖ test) per env]")
    for i in range(len(env_ids)):
        env = f'env{env_ids[i]}'
        kl = kl_divergence(train_cls_cnt[env], test_cls_cnt[env])
        print(f'{env}: {kl:.4f}')


    for i, (tr, va, te) in enumerate(zip(train_dataset, val_dataset, test_dataset)):
        print(f"\tenv{env_ids[i]}:\t{len(tr)}\t|\t{len(va)}\t|\t{len(te)}")

    train_splits, val_splits, test_splits = [], [], []
    train_labels = dict()
    for i, env in enumerate(zip(train_dataset, val_dataset, test_dataset)):
        env_train, env_val, env_test = env
        if hparams['class_balanced']:
            train_weights = misc.make_balanced_weights_per_sample(
                env_train.targets if 'Imbalance' not in args.dataset else env_train.tensors[1].numpy())
            val_weights = misc.make_balanced_weights_per_sample(
                env_val.targets if 'Imbalance' not in args.dataset else env_val.tensors[1].numpy())
            test_weights = misc.make_balanced_weights_per_sample(
                env_test.targets if 'Imbalance' not in args.dataset else env_test.tensors[1].numpy())
        else:
            train_weights, val_weights, test_weights = None, None, None
        train_splits.append((env_train, train_weights))
        val_splits.append((env_val, val_weights))
        test_splits.append((env_test, test_weights))
        train_labels[f"env{env_ids[i]}"] = env_train.targets if 'Imbalance' not in args.dataset else env_train.tensors[1].numpy()

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=num_workers)
        for env, env_weights in train_splits
    ]
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=num_workers)
        for env, _ in (val_splits + test_splits)
    ]
    train_feat_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=num_workers)
        for env, _ in train_splits
    ] if 'BoDA' in args.algorithm else None
    eval_weights = [None for _, weights in (val_splits + test_splits)]
    eval_loader_names = [f'env{env_ids[i]}_val' for i in range(len(val_splits))]
    eval_loader_names += [f'env{env_ids[i]}_test' for i in range(len(test_splits))]
    feat_loader_names = [f'env{env_ids[i]}' for i in range(len(train_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(input_shape, num_classes, len(train_dataset), hparams, env_labels=train_labels)

    if 'CRT' in args.algorithm:
        args.pretrained = os.path.join(
            args.output_dir.replace(args.output_folder_name, args.stage1_folder), hparams['stage1_model']
        ).replace(args.algorithm, args.stage1_algo)
        args.pretrained = args.pretrained.replace(
            f"seed{args.pretrained[args.pretrained.find('seed') + len('seed')]}", 'seed0')
        assert os.path.isfile(args.pretrained)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only = False)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_dict'].items():
            if 'classifier' not in k and 'network.1.' not in k:
                new_state_dict[k] = v
        algorithm.load_state_dict(new_state_dict, strict=False)
        print(f"===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]")
        print(f"===> Pre-trained model loaded: '{args.pretrained}'")

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_step = checkpoint['start_step']
            args.best_val_acc = checkpoint['best_val_acc']
            algorithm.load_state_dict(checkpoint['model_dict'])
            print(f"===> Loaded checkpoint '{args.resume}' (step [{start_step}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    num_domains = len(train_dataset)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env, _ in train_splits])

    def save_checkpoint(best=False, filename='model.pkl', curr_step=0):
        if args.skip_model_save:
            return
        filename = os.path.join(args.output_dir, filename)
        save_dict = {
            "args": vars(args),
            "best_val_acc": args.best_val_acc,
            "start_step": curr_step + 1,
            "num_classes": num_classes,
            "num_domains": len(train_dataset),
            "model_input_shape": input_shape,
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, filename)
        if best:
            shutil.copyfile(filename, filename.replace('pkl', 'best.pkl'))

    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()

        if args.use_meta_learning:
            if args.algorithm != 'LODO_DA_MAML':
                raise ValueError("`--use_meta_learning` is only compatible with `LODO_DA_MAML` algorithm.")

            minibatches_device = [(x.to(device), y.to(device))
                                for x, y in next(train_minibatches_iterator)]
            
            # --- LODO를 위해 for 루프로 모든 도메인을 순회 ---
            all_fold_vals = []
            for i in range(num_domains):
                held_out_domain_idx = i
                
                algorithm.train()
                # 각 도메인이 타겟일 때의 meta-update를 실행
                fold_vals = algorithm.meta_update(
                    minibatches_device,
                    held_out_domain_idx
                )
                all_fold_vals.append(fold_vals)
            
            # --- 각 fold의 결과(손실 등)를 평균내어 이번 스텝의 최종 값으로 사용 ---
            step_vals = {}
            for key in all_fold_vals[0].keys():
                step_vals[key] = np.mean([d[key] for d in all_fold_vals])
        
        # if args.use_meta_learning:
        #     if args.algorithm != 'LODO_DA_MAML':
        #         raise ValueError("`--use_meta_learning` is only compatible with `LODO_DA_MAML` algorithm.")

        #     held_out_domain_idx = random.choice(range(num_domains))
        #     held_out_dataset = train_dataset.datasets[held_out_domain_idx]
            
        #     support, query = create_task_set(held_out_dataset)
            
        #     if support is None:
        #         continue
            
        #     s_x, s_y = support
        #     q_x, q_y = query

        #     s_d = torch.full_like(s_y, held_out_domain_idx)
            
        #     support_set_device = (s_x.to(device), s_y.to(device), s_d.to(device))
        #     query_set_device = (q_x.to(device), q_y.to(device))
            
        #     base_minibatches = [(x.to(device), y.to(device)) for x, y in next(train_minibatches_iterator)]
            
        #     algorithm.train()
        #     step_vals = algorithm.meta_update(
        #         base_minibatches, 
        #         support_set_device, 
        #         query_set_device,
        #         held_out_domain_idx
        #     )
        else:
            minibatches_device = [(x.to(device), y.to(device))
                                for x, y in next(train_minibatches_iterator)]

            train_features = {}
            if 'BoDA' in args.algorithm and (step > 0 and step % hparams["feat_update_freq"] == 0):
                curr_tr_feats, curr_tr_labels = collections.defaultdict(list), collections.defaultdict(list)
                for name, loader in sorted(zip(feat_loader_names, train_feat_loaders), key=lambda x: x[0]):
                    algorithm.eval()
                    with torch.no_grad():
                        for x, y in loader:
                            x, y = x.to(device), y.to(device)
                            feats = algorithm.return_feats(x)
                            curr_tr_feats[name].extend(feats.data)
                            curr_tr_labels[name].extend(y.data)
                train_features = {'feats': curr_tr_feats, 'labels': curr_tr_labels}

            algorithm.train()
            step_vals = algorithm.update(minibatches_device, train_features)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            class_acc_output = collections.defaultdict(list)
            shot_acc_output = collections.defaultdict(list)
            env_acc_output = {}
            for name, loader, weights in sorted(evals, key=lambda x: x[0]):
                if 'test' in name:
                    acc, shot_acc, class_acc = misc.accuracy(
                        algorithm, loader, weights, train_labels[name.split('_')[0]],
                        many_shot_thr, few_shot_thr, device, class_shot_acc=True)
                    class_acc_output[name.split('_')[0]] = list(class_acc)
                    env_acc_output[name.split('_')[0]] = acc
                    shot_acc_output['many'].extend(shot_acc[0])
                    shot_acc_output['median'].extend(shot_acc[1])
                    shot_acc_output['few'].extend(shot_acc[2])
                    shot_acc_output['zero'].extend(shot_acc[3])
                else:
                    acc = misc.accuracy(algorithm, loader, weights, train_labels[name.split('_')[0]],
                                        many_shot_thr, few_shot_thr, device, class_shot_acc=False)
                results[name] = acc

            for shot in ['many', 'median', 'few', 'zero']:
                if len(shot_acc_output[shot]) == 0:
                    shot_acc_output[shot].append(-1)
                results[f"sht_{shot}"] = np.mean(shot_acc_output[shot])

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = list(results.keys())
            if results_keys != last_results_keys:
                print("\n")
                misc.print_row([key for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=8)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=8)

            results.update({
                'hparams': hparams,
                'args': vars(args),
                'class_acc': class_acc_output
            })

            epochs_path = os.path.join(args.output_dir, 'results.json')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            val_env_keys = [f'env{i}_val' for i in env_ids if f'env{i}_val' in results.keys()]
            val_acc_mean = np.mean([results[key] for key in val_env_keys])
            is_best = val_acc_mean > args.best_val_acc
            args.best_val_acc = max(val_acc_mean, args.best_val_acc)
            if is_best:
                best_class_acc = class_acc_output
                best_env_acc = env_acc_output
                best_shot_acc = {s: np.mean(shot_acc_output[s]) for s in ['many', 'median', 'few', 'zero']}

            save_checkpoint(best=is_best, curr_step=step)

            for key in checkpoint_vals.keys() - {'step_time'}:
                tb_logger.log_value(key, results[key], step)
            tb_logger.log_value('val_acc', val_acc_mean, step)
            tb_logger.log_value('test_acc_mean', np.mean(list(env_acc_output.values())), step)
            tb_logger.log_value('test_acc_worst', min(env_acc_output.values()), step)
            for i in env_ids:
                tb_logger.log_value(f'test_env{i}_acc', results[f"env{i}_test"], step)
            for s in ['many', 'median', 'few', 'zero']:
                tb_logger.log_value(f'shot_{s}', results[f"sht_{s}"], step)
            if hasattr(algorithm, 'optimizer'):
                tb_logger.log_value('learning_rate', algorithm.optimizer.param_groups[0]['lr'], step)

            checkpoint_vals = collections.defaultdict(lambda: [])

    print("\nTest accuracy (best validation checkpoint):")
    print(f"\tmean:\t[{np.mean(list(best_env_acc.values())):.3f}]\n\tworst:\t[{min(best_env_acc.values()):.3f}]")
    print("Shot-wise accuracy:")
    for s in ['many', 'median', 'few', 'zero']:
        print(f"\t[{s[:4]}]:\t[{best_shot_acc[s]:.3f}]")
    print("Class-wise accuracy:")
    for env in sorted(best_class_acc):
        print('\t[{}] overall {:.3f}, class-wise {}'.format(
            env, best_env_acc[env], (np.array2string(
                np.array(best_class_acc[env]), separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

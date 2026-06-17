from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import logging
import argparse
import shutil
from collections import defaultdict

import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
from dataloader.dataloader import BaseDataloader,SubsetTrainDataloader,LABELNAMES
from networks.models import DenseNet121
from utils.Test import epoch_metrics
from utils import losses
from utils.MSC import MultiLevelSemanticCorrector,SoftGoldEvaluator
from utils.ASC import ASCBatchSampler, compute_clean_count_from_gmm_dataset
parser = argparse.ArgumentParser(description='PyTorch ISIC Training')

parser.add_argument('--num_class', default=9, type=int)
parser.add_argument('--num_train', default=70000, type=int)#
parser.add_argument('--dataset', default='NCTCRCHE', type=str)
parser.add_argument('--dataset_csv_path',  default='/data/NCTCRCHE/', type=str)
parser.add_argument('--dataset_root_path', default='/data/NCTCRCHE/', type=str, help='path to dataset')
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')

parser.add_argument('--seed', default=7900, type=int)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--num_warmup', default=5, type=int)
parser.add_argument('--gpu', type=str,  default='1,2', help='GPU to use')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='initial learning rate')

parser.add_argument('--noise_mode',  default='gammasym',help='gammasym/asym/sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.75, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--id', default='')
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--exp', type=str,  default='test', help='model_name')
parser.add_argument('--majority_classes', default='6,7', type=str)
parser.add_argument('--minority_classes', default='5,8', type=str)
parser.add_argument('--middle_classes', default='0,1,2,3,4', type=str)

parser.add_argument('--msc_conf_ema', default=0.75, type=float)
parser.add_argument('--msc_k_list', default='', type=str)
parser.add_argument('--msc_topk', default=-1, type=int)
parser.add_argument('--msc_queue_len', default=-1, type=int)
parser.add_argument('--msc_update_every', default=1, type=int)
parser.add_argument('--asc_anchor_gamma', default=1.0, type=float)
args = parser.parse_args()
if args.majority_classes.strip() != '':
    args.majority_classes = [int(x) for x in args.majority_classes.split(',') if x.strip() != '']
else:
    args.majority_classes = []
if args.minority_classes.strip() != '':
    args.minority_classes = [int(x) for x in args.minority_classes.split(',') if x.strip() != '']
else:
    args.minority_classes = []

if args.middle_classes.strip() != '':
    args.middle_classes = [int(x) for x in args.middle_classes.split(',') if x.strip() != '']
else:
    args.middle_classes = []

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
best_Accus_avg=0
best_epoch=0
batch_size = args.batch_size * len(args.gpu.split(','))
cudnn.benchmark = False
cudnn.deterministic = True
torch.cuda.manual_seed(args.seed)

def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, msc_self, SGEvaluator):
    net.train()
    net2.eval()

    def _sharpen(p, T, eps=1e-12):
        p = torch.clamp(p, min=eps)
        p = p ** (1.0 / T)
        return p / torch.clamp(p.sum(dim=1, keepdim=True), min=eps)

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = len(labeled_trainloader)

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, golds_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, golds_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, golds_u = next(unlabeled_train_iter)

        bs = inputs_x.size(0)

        labels_x_oh = torch.zeros(bs, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2 = inputs_x.cuda(), inputs_x2.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
        labels_x_oh, w_x = labels_x_oh.cuda(), w_x.cuda()

        net_was_training = net.training
        net.eval()
        net2.eval()

        with torch.no_grad():
            feat_u11, out_u11 = net(inputs_u)
            feat_u12, out_u12 = net(inputs_u2)
            feat_u21, out_u21 = net2(inputs_u)
            feat_u22, out_u22 = net2(inputs_u2)

            # 4-prob co-guess base
            pu = (torch.softmax(out_u11, dim=1) + torch.softmax(out_u12, dim=1) +
                  torch.softmax(out_u21, dim=1) + torch.softmax(out_u22, dim=1)) / 4.0
            anchor_u = pu.argmax(dim=1)
            feat_u = (feat_u11 + feat_u12) / 2.0
            p_msc_u = msc_self.build_soft_labels_noisy(feat_u, anchor_u)
            p_cross_u = 0.5 * (p_msc_u + pu)
            targets_u = _sharpen(p_cross_u, T=args.T)

            feat_x1, out_x1 = net(inputs_x)
            feat_x2, out_x2 = net(inputs_x2)
            feat_x = (feat_x1 + feat_x2) / 2.0

            noisy_cls_x = labels_x_oh.argmax(dim=1)
            p_msc_corr_x = msc_self.build_soft_labels_clean_corr_only(feat_x, noisy_cls_x)

            _, out_x21 = net2(inputs_x)
            _, out_x22 = net2(inputs_x2)

            px = (torch.softmax(out_x1, dim=1) + torch.softmax(out_x2, dim=1) +
                  torch.softmax(out_x21, dim=1) + torch.softmax(out_x22, dim=1)) / 4.0

            p_cross_x = 0.5 * (p_msc_corr_x + px)
            p_cross_x = _sharpen(p_cross_x, T=args.T)

            targets_x = w_x * labels_x_oh + (1.0 - w_x) * p_cross_x
            targets_x = targets_x / torch.clamp(targets_x.sum(dim=1, keepdim=True), min=1e-12)

            msc_self.update_queues_if_clean(feat_x, noisy_cls_x, targets_x)

            SGEvaluator.update_x(noisy_x=labels_x, soft_x=targets_x, gold_x=golds_x)
            SGEvaluator.update_u(pu=pu, soft_u=targets_u, gold_u=golds_u)

        if net_was_training:
            net.train()
        else:
            net.eval()
        net2.eval()

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        _, logits = net(mixed_input)
        logits_x = logits[:bs * 2]
        logits_u = logits[bs * 2:]

        Lx, Lu, lamb = criterion(
            logits_x, mixed_target[:bs * 2],
            logits_u, mixed_target[bs * 2:],
            epoch + batch_idx / num_iter,
            args.num_warmup
        )

        prior = torch.ones(args.num_class).cuda() / args.num_class
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            '%s:Epoch [%3d/%3d] Iter[%3d/%3d] Lx: %.4f Lu: %.4f'
            % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter, Lx.item(), Lu.item())
        )
        sys.stdout.flush()

def warmup(epoch, net1, net2, optimizer1, optimizer2, dataloader):
    net1.train()
    net2.train()
    num_iter = len(dataloader)

    for batch_idx, (inputs, _, _, targets, _, _, image_name) in enumerate(dataloader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # targets -> hard label
        if targets.dim() > 1:
            targets_cls = targets.argmax(dim=1)
        else:
            targets_cls = targets
        targets_cls = targets_cls.long()

        # ---- update net1 ----
        _, outputs1 = net1(inputs)
        # loss1 = CEloss(outputs1, targets_cls)
        loss1= loss_mean(outputs1, targets) 
        optimizer1.zero_grad(set_to_none=True)
        loss1.backward()
        optimizer1.step()

        # ---- update net2 ----
        _, outputs2 = net2(inputs)
        # loss2 = CEloss(outputs2, targets_cls)
        loss2= loss_mean(outputs2, targets) 
        optimizer2.zero_grad(set_to_none=True)
        loss2.backward()
        optimizer2.step()

        # logging
        sys.stdout.write('\r')
        sys.stdout.write(
            '%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss1: %.4f  loss2: %.4f'
            % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs,
               batch_idx + 1, num_iter, loss1.item(), loss2.item())
        )
        sys.stdout.flush()
   
   
   
def GMM_train(args, model1, model2, all_loss1, all_loss2, gmm_loader):
    model1.eval()
    model2.eval()

    N = args.num_train

    # ---- buffers for GMM loss computation (GPU) ----
    losses1 = torch.zeros(N, device='cuda', dtype=torch.float32)
    losses2 = torch.zeros(N, device='cuda', dtype=torch.float32)
    ys      = torch.zeros(N, device='cuda', dtype=torch.long)  # noisy hard label

    # ---- buffers for MSC update (CPU) ----
    feat1_buf = None
    feat2_buf = None
    preds1_buf = torch.empty(N, device='cpu', dtype=torch.long)
    preds2_buf = torch.empty(N, device='cpu', dtype=torch.long)

    with torch.no_grad():
        for batch_idx, (inputs, _, _, targets, index, _, _) in enumerate(gmm_loader):
            index_cpu = index.long()  # CPU index for cache write
            inputs  = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # targets -> hard class id
            if targets.dim() > 1:
                targets_cls = targets.argmax(dim=1)
            else:
                targets_cls = targets
            targets_cls = targets_cls.long()

            # forward both models on the same batch
            features1, outputs1 = model1(inputs)
            features2, outputs2 = model2(inputs)

            # per-sample loss for both models (CE must be reduction='none')
            loss1 = CE(outputs1, targets_cls)  # [B]
            loss2 = CE(outputs2, targets_cls)  # [B]

            # write to global buffers by index (GPU for loss/ys)
            index_gpu = index_cpu.cuda(non_blocking=True)
            losses1[index_gpu] = loss1.detach()
            losses2[index_gpu] = loss2.detach()
            ys[index_gpu]      = targets_cls.detach()

            # allocate feature buffers after first forward
            if feat1_buf is None:
                feat_dim = int(features1.shape[1])
                feat1_buf = torch.empty((N, feat_dim), device='cpu', dtype=torch.float16)
                feat2_buf = torch.empty((N, feat_dim), device='cpu', dtype=torch.float16)

            # write caches (CPU)
            feat1_buf[index_cpu]  = features1.detach().to('cpu', dtype=torch.float16)
            feat2_buf[index_cpu]  = features2.detach().to('cpu', dtype=torch.float16)
            preds1_buf[index_cpu] = outputs1.argmax(dim=1).detach().to('cpu', dtype=torch.long)
            preds2_buf[index_cpu] = outputs2.argmax(dim=1).detach().to('cpu', dtype=torch.long)

    def _loss_to_prob_and_history(losses_gpu, ys_gpu, all_loss_hist):
        # per-class z-score normalize on GPU
        losses_norm = torch.empty_like(losses_gpu)
        for c in range(args.num_class):
            m = (ys_gpu == c)
            if int(m.sum().item()) < 2:
                losses_norm[m] = losses_gpu[m]
                continue
            lc = losses_gpu[m]
            mu = lc.mean()
            sd = lc.std(unbiased=False) + 1e-6
            losses_norm[m] = (lc - mu) / sd

        losses_norm = torch.clamp(losses_norm, -5.0, 5.0)

        # store history on CPU
        losses_norm_cpu = losses_norm.detach().cpu()
        all_loss_hist.append(losses_norm_cpu)

        # build GMM input on CPU
        if args.r == 0.9:
            history = torch.stack(all_loss_hist, dim=0)  # [T, N]
            k = min(5, history.size(0))
            input_loss_t = history[-k:].mean(0)          # [N] on CPU
        else:
            input_loss_t = losses_norm_cpu

        input_loss_np = input_loss_t.reshape(-1, 1).numpy().astype(np.float32)

        # fit GMM on CPU
        gmm = GaussianMixture(
            n_components=2,
            max_iter=10,
            tol=1e-2,
            reg_covar=5e-4,
            random_state=args.seed
        )
        gmm.fit(input_loss_np)

        prob_np = gmm.predict_proba(input_loss_np)
        clean_comp = int(np.argmin(gmm.means_.reshape(-1)))
        prob = prob_np[:, clean_comp]  # [N]
        return prob, all_loss_hist

    # fit two GMMs (one per model)
    prob1, all_loss1 = _loss_to_prob_and_history(losses1, ys, all_loss1)
    prob2, all_loss2 = _loss_to_prob_and_history(losses2, ys, all_loss2)

    cache1 = {
        "features": feat1_buf,               # CPU [N, D], float16
        "targets": ys.detach().cpu().long(), # CPU [N]
        "preds": preds1_buf                  # CPU [N]
    }
    cache2 = {
        "features": feat2_buf,
        "targets": ys.detach().cpu().long(),
        "preds": preds2_buf
    }

    return prob1, prob2, all_loss1, all_loss2, cache1, cache2



class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    # Network definition
    net = DenseNet121(out_size=args.num_class, mode=args.label_uncertainty, drop_rate=args.drop_rate)
    if len(args.gpu.split(',')) > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    return model

def safe_split(pred, prob, min_labeled=batch_size, min_unlabeled=batch_size):
    N = len(pred)
    labeled_idx   = np.where(pred)[0]
    unlabeled_idx = np.where(~pred)[0]

    if len(labeled_idx) < min_labeled:
        print("Labeled set is empty; applying minimum fallback.")
        sorted_idx = np.argsort(-prob)
        labeled_idx = sorted_idx[:min_labeled]
        unlabeled_idx = np.setdiff1d(np.arange(N), labeled_idx)

    if len(unlabeled_idx) < min_unlabeled:
        print("Unlabeled set is empty; applying minimum fallback.")
        sorted_idx = np.argsort(prob)
        unlabeled_idx = sorted_idx[:min_unlabeled]
        labeled_idx = np.setdiff1d(np.arange(N), unlabeled_idx)

    return labeled_idx, unlabeled_idx

def idx_class_clean_noise_stats(GMM_dataset, indices, num_class=9):
    stats = {
        "total": np.zeros(num_class, dtype=int),
        "clean": np.zeros(num_class, dtype=int),
        "noisy": np.zeros(num_class, dtype=int),
    }

    for j in indices:
        _, _, _,target, _, gold,_ = GMM_dataset[j]

        y_gold  = int(np.argmax(gold))
        y_noisy = int(np.argmax(target))

        stats["total"][y_gold] += 1
        if y_gold == y_noisy:
            stats["clean"][y_gold] += 1
        else:
            stats["noisy"][y_gold] += 1

    return stats

snapshot_path = "../model/" + args.exp + "/"
if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    lr_=args.lr
    noise_file_path=os.path.join(args.dataset_csv_path,f'NoisyLabels_r{args.r}_seed{args.seed}.csv') 
    base_loader = BaseDataloader (dataset=args.dataset,r=args.r,batch_size= batch_size,num_workers=8,\
        root_dir=args.dataset_root_path,num_class=args.num_class,dataset_csv_path=args.dataset_csv_path,\
        seed=args.seed,noise_file=noise_file_path)

    subsetTrain_dataloader=SubsetTrainDataloader (batch_size=batch_size,num_workers=8)
    print('| Building net')
    net1 = create_model()
    net2 = create_model()

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).cuda()
        feat_dim = net1(dummy)[0].shape[1]

    if args.msc_k_list.strip() == '':
        k_list = []
    else:
        k_list = [int(x) for x in args.msc_k_list.split(',') if x.strip() != '']

    msc_topk = args.msc_topk if args.msc_topk > 0 else batch_size
    msc_queue_len = args.msc_queue_len if args.msc_queue_len > 0 else batch_size

    msc1 = MultiLevelSemanticCorrector(
        num_class=args.num_class,
        feat_dim=feat_dim,
        queue_len=msc_queue_len,
        topk=msc_topk,
        conf_ema=args.msc_conf_ema,
        k_list=k_list,
        majority_classes=args.majority_classes,
        minority_classes=args.minority_classes,
        middle_classes=args.middle_classes
    )

    msc2 = MultiLevelSemanticCorrector(
        num_class=args.num_class,
        feat_dim=feat_dim,
        queue_len=msc_queue_len,
        topk=msc_topk,
        conf_ema=args.msc_conf_ema,
        k_list=k_list,
        majority_classes=args.majority_classes,
        minority_classes=args.minority_classes,
        middle_classes=args.middle_classes
    )
    SGEvaluator = SoftGoldEvaluator(num_class=args.num_class, topk=(5,))
    optimizer1= torch.optim.Adam(net1.parameters(), lr=lr_, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=lr_, betas=(0.9, 0.999), weight_decay=5e-4)

    loss_mean = losses.cross_entropy_loss(reduction='mean')
    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode!='sym' :
        conf_penalty = NegEntropy()
    criterion = losses.SemiLoss(args.lambda_u)

    all_loss = [[],[]] # save the history of losses from two networks

    test_loader = base_loader .run('test')
    print('| Success: test dataloader built')
    GMM_loader,GMM_dataset = base_loader .run('GMM_train')   
    print('| Success: GMM_train dataloader built')
    warmup_trainloader = base_loader .run('warmup')
    print('| Success: warmup dataloader built')
    if args.resume:
        assert os.path.isfile(args.resume), f"=> no checkpoint found at '{args.resume}'"
        logging.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)

        args.start_epoch = checkpoint['epoch']
        epoch= args.start_epoch 
        net1.load_state_dict(checkpoint['net1_state_dict'])
        net2.load_state_dict(checkpoint['net2_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])

        lr_= args.lr * (0.9 ** args.start_epoch)  
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr_     
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr_

        if epoch>args.num_warmup:
            best_epoch = checkpoint.get('best_epoch', 0)
            best_AUROCs = checkpoint.get('best_AUROCs', None)
            best_Accus = checkpoint.get('best_Accus', None)
            best_Senss = checkpoint.get('best_Senss', None)
            best_Specs = checkpoint.get('best_Specs', None)
            best_AUROC_avg = checkpoint.get('best_AUROC_avg', 0)
            best_Accus_avg = checkpoint.get('best_Accus_avg', 0)
            best_Senss_avg = checkpoint.get('best_Senss_avg', 0)
            best_Specs_avg = checkpoint.get('best_Specs_avg', 0)

        if 'all_loss' in checkpoint:
            all_loss = checkpoint['all_loss']
        else:
            all_loss = [[], []]

        if 'rng_state' in checkpoint:
            random.setstate(checkpoint['rng_state']['random'])
            np.random.set_state(checkpoint['rng_state']['np'])
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
            logging.info("=> restored RNG states.")

        logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    for epoch in range(args.start_epoch,args.num_epochs+1):   
        if epoch<args.num_warmup:  
            print('Warmup Net')
            warmup(epoch, net1, net2, optimizer1, optimizer2, warmup_trainloader)

            if (epoch % args.msc_update_every) == 0:
                msc1.epoch_update_from_dataloader(net1, GMM_loader, clean_index_set=None, proto_ema_beta=0.9)
                msc1.build_class_groups()
                msc1.print_class_groups(epoch)

                msc2.epoch_update_from_dataloader(net2, GMM_loader, clean_index_set=None, proto_ema_beta=0.9)
                msc2.build_class_groups()
                msc2.print_class_groups(epoch)
        else:  
            prob1, prob2, all_loss[0], all_loss[1], cache1, cache2 = GMM_train(args, net1, net2,\
                                                                        all_loss[0], all_loss[1], GMM_loader)
            pred1 = (prob1 > args.p_threshold)      
            pred2 = (prob2 > args.p_threshold)      

            print('Train Net1')
            # ===== Train Net1 =====
            labeled_idx2, unlabeled_idx2 = safe_split(pred2, prob2)

            if (epoch % args.msc_update_every) == 0:
                msc1.epoch_update_from_dataloader(
                    net1, GMM_loader,
                    clean_index_set=set(labeled_idx2.tolist()),
                    proto_ema_beta=0.9,
                    cache=cache1
                )
                msc1.build_class_groups()
                msc1.print_class_groups(epoch)

            labeled_trainloader, unlabeled_trainloader = subsetTrain_dataloader.run(
                args.dataset_root_path, GMM_dataset, args.dataset_csv_path,
                labeled_idx2, unlabeled_idx2, prob2
            )
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, msc1, SGEvaluator)

                
            print('\nTrain Net2')
            # ===== Train Net2 =====
            labeled_idx1, unlabeled_idx1 = safe_split(pred1, prob1)

            if (epoch % args.msc_update_every) == 0:
                msc2.epoch_update_from_dataloader(
                    net2, GMM_loader,
                    clean_index_set=set(labeled_idx1.tolist()),
                    proto_ema_beta=0.9,
                    cache=cache2
                )
                msc2.build_class_groups()
                msc2.print_class_groups(epoch)

            clean_count = compute_clean_count_from_gmm_dataset(
                GMM_dataset, labeled_idx1, args.num_class
            )

            asc_batch_sampler = ASCBatchSampler(
                global_indices=labeled_idx1,
                GMM_dataset=GMM_dataset,
                num_class=args.num_class,
                batch_size=batch_size,                  
                class_groups=msc2.class_groups,         
                majority_classes=args.majority_classes, 
                clean_count=clean_count,
                anchor_gamma=args.asc_anchor_gamma,             
                drop_last=True,
                seed=args.seed + epoch                
            )

            labeled_trainloader, unlabeled_trainloader = subsetTrain_dataloader.run(
                args.dataset_root_path, GMM_dataset, args.dataset_csv_path,
                labeled_idx1, unlabeled_idx1, prob1,
                labeled_batch_sampler=asc_batch_sampler
            )

            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, msc2, SGEvaluator)
            
            SGEvaluator.pretty_print(epoch)
        AUROCs, Accus, Senss, Specs, F1s, MCCs, Macro_F1, Macro_MCC, TPs, FPs, FNs, TNs = epoch_metrics(
                        net1, net2, test_loader, LABELNAMES)
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()

        F1_macro  = Macro_F1
        MCC_macro = Macro_MCC

        logging.info("\nTEST Student: Epoch: {}".format(epoch))
        logging.info(
            "\nTEST Accus: {:.6f}, Senss: {:.6f}, Specs: {:.6f}, AUROC: {:.6f}, Macro-F1: {:.6f}, Macro-MCC: {:.6f}".format(
                Accus_avg, Senss_avg, Specs_avg, AUROC_avg, F1_macro, MCC_macro
            )
        )

        logging.info("Accus:  " + " ".join(["{}:{:.6f}".format(LABELNAMES[i], v) for i, v in enumerate(Accus)]))
        logging.info("Senss:  " + " ".join(["{}:{:.6f}".format(LABELNAMES[i], v) for i, v in enumerate(Senss)]))
        logging.info("Specs:  " + " ".join(["{}:{:.6f}".format(LABELNAMES[i], v) for i, v in enumerate(Specs)]))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(LABELNAMES[i], v) for i, v in enumerate(AUROCs)]))

        logging.info("F1s:    " + " ".join(["{}:{:.6f}".format(LABELNAMES[i], v) for i, v in enumerate(F1s)]))
        logging.info("MCCs:   " + " ".join(["{}:{:.6f}".format(LABELNAMES[i], v) for i, v in enumerate(MCCs)]))
        logging.info("TPs:    " + " ".join(["{}:{}".format(LABELNAMES[i], v) for i, v in enumerate(TPs)]))
        logging.info("FPs:    " + " ".join(["{}:{}".format(LABELNAMES[i], v) for i, v in enumerate(FPs)]))
        logging.info("FNs:    " + " ".join(["{}:{}".format(LABELNAMES[i], v) for i, v in enumerate(FNs)]))
        logging.info("TNs:    " + " ".join(["{}:{}".format(LABELNAMES[i], v) for i, v in enumerate(TNs)]))
        Ps = [TPs[i] + FNs[i] for i in range(len(TPs))]
        Ns = [TNs[i] + FPs[i] for i in range(len(TPs))]
        logging.info("PosCnt: " + " ".join(["{}:{}".format(LABELNAMES[i], v) for i,v in enumerate(Ps)]))
        logging.info("NegCnt: " + " ".join(["{}:{}".format(LABELNAMES[i], v) for i,v in enumerate(Ns)]))
       
       
        if epoch>args.num_warmup:
            if Accus_avg>best_Accus_avg:
                best_epoch=epoch
                best_AUROCs, best_Accus, best_Senss, best_Specs=AUROCs, Accus, Senss, Specs
                best_AUROC_avg =AUROC_avg 
                best_Accus_avg = Accus_avg
                best_Senss_avg = Senss_avg 
                best_Specs_avg = Specs_avg 

            logging.info("\nbest_Student: best_epoch: {}".format(best_epoch))
            logging.info("\nbest_Accus: {:6f}, best_Senss: {:6f}, best_Specs: {:6f}, best_AUROC: {:6f}"
                        .format(best_Accus_avg, best_Senss_avg, best_Specs_avg,best_AUROC_avg ))

        save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch+1) + '.pth')
        if epoch>args.num_warmup:
            torch.save({
                'epoch': epoch + 1,
                'all_loss': all_loss,
                'net1_state_dict': net1.state_dict(),
                'net2_state_dict': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'best_epoch': best_epoch,
                'best_AUROCs': best_AUROCs,
                'best_Accus': best_Accus,
                'best_Senss': best_Senss,
                'best_Specs': best_Specs,
                'best_AUROC_avg': best_AUROC_avg,
                'best_Accus_avg': best_Accus_avg,
                'best_Senss_avg': best_Senss_avg,
                'best_Specs_avg': best_Specs_avg,
                'rng_state': {
                    'random': random.getstate(),
                    'np': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all()
                },
            }, save_mode_path)
        else:
            torch.save({
                'epoch': epoch + 1,
                'all_loss': all_loss,
                'net1_state_dict': net1.state_dict(),
                'net2_state_dict': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'rng_state': {
                    'random': random.getstate(),
                    'np': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all()
                },
            }, save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

        lr_ = lr_ * 0.9
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr_       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr_

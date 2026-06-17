import torch
import numpy as np


import torch
import numpy as np
#
class MultiLevelSemanticCorrectorOnlyProto(object):

    def __init__(self, num_class, feat_dim, queue_len, topk, conf_ema=0.75,
                 k_list=None, majority_classes=None, minority_classes=None, middle_classes=None,
                 device='cuda'):
        self.num_class = int(num_class)
        self.feat_dim = int(feat_dim)
        self.device = device

        self.queue_len = int(queue_len)
        self.topk = int(topk)

        self.conf_ema = float(conf_ema)

        if majority_classes is None:
            self.majority_classes = set()
        else:
            if isinstance(majority_classes, (int, np.integer)):
                majority_classes = [int(majority_classes)]
            self.majority_classes = set(int(x) for x in list(majority_classes))

        if minority_classes is None:
            self.minority_classes = set()
        else:
            if isinstance(minority_classes, (int, np.integer)):
                minority_classes = [int(minority_classes)]
            self.minority_classes = set(int(x) for x in list(minority_classes))

        if middle_classes is None:
            self.middle_classes = set()
        else:
            if isinstance(middle_classes, (int, np.integer)):
                middle_classes = [int(middle_classes)]
            self.middle_classes = set(int(x) for x in list(middle_classes))

        if k_list is None or len(k_list) == 0:
            self.k_list = [max(2, min(self.num_class - 1, self.num_class // 2)) for _ in range(self.num_class)]
        else:
            assert len(k_list) == self.num_class, "msc_k_list 必须提供 num_class 个整数"
            self.k_list = [max(2, min(self.num_class - 1, int(k))) for k in k_list]

        self.prototypes = torch.zeros(self.num_class, self.feat_dim, device=self.device)

        self.confusion_ema = torch.zeros(self.num_class, self.num_class, device='cpu', dtype=torch.float32)

        self.class_groups = {a: [] for a in range(self.num_class)}

    def _l2norm(self, x, eps=1e-12):
        return x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)

    def _one_hot(self, cls_idx):
        # cls_idx: [B]
        B = cls_idx.size(0)
        y = torch.zeros(B, self.num_class, device=cls_idx.device, dtype=torch.float32)
        y.scatter_(1, cls_idx.view(-1, 1), 1.0)
        return y

    def _safe_softmax(self, logits, dim=-1):

        m = logits.max(dim=dim, keepdim=True).values
        ex = torch.exp(logits - m)
        return ex / torch.clamp(ex.sum(dim=dim, keepdim=True), min=1e-12)


    @torch.no_grad()
    def epoch_update_from_dataloader(
        self,
        net,
        loader,
        clean_index_set=None,
        proto_ema_beta=0.9,
        cache=None,
    ):
        
        net.eval()

        
        sum_feat = torch.zeros(self.num_class, self.feat_dim, device=self.device)
        cnt_feat = torch.zeros(self.num_class, device=self.device)

      
        conf_cnt = torch.zeros(self.num_class, self.num_class, device="cpu", dtype=torch.float32)

        def _pick_inputs(batch):
     
            if isinstance(batch, (list, tuple)):
                for x in batch:
                    if torch.is_tensor(x) and x.ndim == 4:
                        return x
       
                return batch[0]
            return batch

        def _pick_labels_and_index(batch, B):
            labels = None
            index = None

            if not isinstance(batch, (list, tuple)):
                return labels, index

       
            candidates = []
            for item in batch:
                if not torch.is_tensor(item):
                    continue
                if item.ndim == 0:
                    continue
                if item.size(0) != B:
                    continue
                candidates.append(item)

       
            for t in candidates:
                if t.ndim == 1:
                    labels = t
                    break
                if t.ndim == 2 and t.size(1) == self.num_class:
                    labels = t
                    break

            for t in candidates:
                if labels is not None and t is labels:
                    continue
                if t.ndim == 1:
                    index = t

                    break

            return labels, index

        def _normalize_labels(labels, device):
            
            labels = labels.to(device, non_blocking=True)

            if labels.ndim == 2 and labels.size(1) == self.num_class:
                labels = torch.argmax(labels, dim=1)
            elif labels.ndim > 1:
                labels = labels.view(-1)

            return labels.long()

        for batch in loader:
            # -------- inputs --------
            inputs = _pick_inputs(batch)
            if not torch.is_tensor(inputs):
                raise RuntimeError("epoch_update_from_dataloader: batch 中未找到可用的 inputs 张量")
            inputs = inputs.to(self.device, non_blocking=True)
            B = int(inputs.size(0))

            # -------- labels / index --------
            labels, index = _pick_labels_and_index(batch, B)
            if labels is None:
               
                raise RuntimeError(
                    "epoch_update_from_dataloader: 找不到 labels。请打印一个 batch 的各元素 shape/dtype 以确认结构。"
                )
            labels = _normalize_labels(labels, self.device)

            # -------- forward --------
            out = net(inputs)

            
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                feat, logits = out[0], out[1]
            elif isinstance(out, dict) and ("feat" in out and "logits" in out):
                feat, logits = out["feat"], out["logits"]
            else:
                raise RuntimeError("epoch_update_from_dataloader: net(inputs) 需要返回 (feat, logits) 或包含 feat/logits 的 dict")

            feat = self._l2norm(feat)
            pred = torch.argmax(logits, dim=1)

  
            pred_cpu = pred.detach().cpu().tolist()
            lab_cpu = labels.detach().cpu().tolist()
            for a, j in zip(pred_cpu, lab_cpu):
                if 0 <= int(a) < self.num_class and 0 <= int(j) < self.num_class:
                    conf_cnt[int(a), int(j)] += 1.0

            if clean_index_set is not None and index is not None:
                # 将 index 转成 python list
                if torch.is_tensor(index):
                    idx_list = index.detach().cpu().view(-1).tolist()
                else:
                    idx_list = list(index)

                # clean mask
                mask = torch.tensor([i in clean_index_set for i in idx_list], device=self.device, dtype=torch.bool)

                if mask.any():
                    feat_c = feat[mask]
                    lab_c = labels[mask]
                    for c in range(self.num_class):
                        m = (lab_c == c)
                        if m.any():
                            sum_feat[c] += feat_c[m].sum(dim=0)
                            cnt_feat[c] += float(m.sum().item())
            else:
            
                for c in range(self.num_class):
                    m = (labels == c)  # (B,)
                    if m.any():
                        sum_feat[c] += feat[m].sum(dim=0)
                        cnt_feat[c] += float(m.sum().item())

      
        for c in range(self.num_class):
            if cnt_feat[c] > 0:
                new_p = sum_feat[c] / cnt_feat[c]
                new_p = self._l2norm(new_p)
                self.prototypes[c] = proto_ema_beta * self.prototypes[c] + (1.0 - proto_ema_beta) * new_p
                self.prototypes[c] = self._l2norm(self.prototypes[c])


        self.confusion_ema = self.conf_ema * self.confusion_ema + (1.0 - self.conf_ema) * conf_cnt

        net.train()



    @torch.no_grad()
    def build_class_groups(self):
        
        P = self._l2norm(self.prototypes)  
        sim = torch.matmul(P, P.t()).detach().cpu()  

        for a in range(self.num_class):
            K = int(self.k_list[a])
            K = max(2, min(self.num_class - 1, K))

            
            conf_row = self.confusion_ema[a].clone()
            conf_row[a] = -1.0
            conf_rank = torch.argsort(conf_row, descending=True).tolist()

           
            sim_row = sim[a].clone()
            sim_row[a] = -1.0
            sim_rank = torch.argsort(sim_row, descending=True).tolist()

            
            half = max(1, K // 2)
            conf_top = [j for j in conf_rank if j != a][:K]
            sim_top = [j for j in sim_rank if j != a][:K]

            inter = []
            sset = set(sim_top[:half])
            for j in conf_top[:half]:
                if j in sset:
                    inter.append(j)

            g = inter.copy()
            for j in conf_top:
                if len(g) >= K:
                    break
                if j == a:
                    continue
             
                if j in self.majority_classes:
                    continue
                if j not in g:
                    g.append(j)

           
            for j in sim_top:
                if len(g) >= K:
                    break
                if j == a or j in self.majority_classes:
                    continue
                if j not in g:
                    g.append(j)

            self.class_groups[a] = g

    def print_class_groups(self, epoch):
        print(f"[MSC][epoch={epoch}] class_groups:")
        for a in range(self.num_class):
            print(f"[MSC][epoch={epoch}] g[{a}]={self.class_groups.get(a, [])}")


    @torch.no_grad()
    def _class_level_soft(self, features, anchor_cls, temperature=1.0):
       
        feats = self._l2norm(features)
        B = feats.size(0)

      
        proto_norm = torch.norm(self.prototypes, dim=1)  # [C]
        if (proto_norm < 1e-6).all():
            return self._one_hot(anchor_cls)

        P = self._l2norm(self.prototypes)  # [C, D]
        out = torch.zeros(B, self.num_class, device=feats.device, dtype=torch.float32)

        for i in range(B):
            a = int(anchor_cls[i].item())
            cand = [a] + list(self.class_groups.get(a, []))
            
            cand = [c for c in dict.fromkeys(cand) if 0 <= int(c) < self.num_class]

           
            pv = P[cand]  # [M, D]
            logits = torch.matmul(pv, feats[i]) / max(1e-6, float(temperature))  # [M]
            prob = self._safe_softmax(logits, dim=0)

            
            for k, c in enumerate(cand):
                out[i, int(c)] = prob[k]


        out = out / torch.clamp(out.sum(dim=1, keepdim=True), min=1e-12)
        return out

    @torch.no_grad()
    def build_soft_labels_clean_corr_only(self, features, noisy_cls):
        return self._class_level_soft(features, noisy_cls)

    @torch.no_grad()
    def build_soft_labels_noisy(self, features, anchor_cls):
        return self._class_level_soft(features, anchor_cls)

    @torch.no_grad()
    def build_soft_labels_clean(self, features, noisy_cls, alpha):
        return self._class_level_soft(features, noisy_cls)

    @torch.no_grad()
    def update_queues_if_clean(self, feat_x, noisy_cls_x, soft_x):
        return

class MultiLevelSemanticCorrector(object):

    def __init__(self, num_class, feat_dim, queue_len, topk, conf_ema=0.75,
                 k_list=None, majority_classes=None, minority_classes=None, middle_classes=None,device='cuda'):  
        
        self.num_class = int(num_class)  
        self.feat_dim = int(feat_dim)    
        self.device = device             

        self.queue_len = int(queue_len)  
        self.topk = int(topk)           

        self.conf_ema = float(conf_ema)  
      
        if majority_classes is None:
            self.majority_classes = set()
        else:
          
            if isinstance(majority_classes, (int, np.integer)):
                majority_classes = [int(majority_classes)]
            self.majority_classes = set(int(x) for x in list(majority_classes))
               
        if minority_classes is None:
            self.minority_classes = set()
        else:
            if isinstance(minority_classes, (int, np.integer)):
                minority_classes = [int(minority_classes)]
            self.minority_classes = set(int(x) for x in list(minority_classes))

        if middle_classes is None:
            self.middle_classes = set()
        else:
            if isinstance(middle_classes, (int, np.integer)):
                middle_classes = [int(middle_classes)]
            self.middle_classes = set(int(x) for x in list(middle_classes))

        if k_list is None or len(k_list) == 0:  
            self.k_list = [max(2, min(self.num_class - 1, self.num_class // 2)) for _ in range(self.num_class)]  
        else: 
            assert len(k_list) == self.num_class, "msc_k_list 必须提供 num_class 个整数" 
            self.k_list = [max(2, min(self.num_class - 1, int(k))) for k in k_list]  

        self.prototypes = torch.zeros(self.num_class, self.feat_dim, device=self.device)  

        self.confusion_ema = torch.zeros(self.num_class, self.num_class, device='cpu', dtype=torch.float32)  

       
        self.class_groups = {a: [] for a in range(self.num_class)} 

        self.queues_feat = {c: [] for c in range(self.num_class)}  

        self.queues_soft = {c: [] for c in range(self.num_class)}  

    def _l2norm(self, x, eps=1e-12):
        
        return x / (x.norm(dim=-1, keepdim=True) + eps)  

    def _cosine_sim(self, a, b, eps=1e-12):
        
        a = self._l2norm(a, eps)                
        b = self._l2norm(b, eps)                
        return torch.mm(a, b.t())               

    def _normalize_full(self, y, eps=1e-12):
        
        if y.dim() == 1:                        
            s = y.sum().clamp(min=eps)            
            return y / s                         
        s = y.sum(dim=1, keepdim=True).clamp(min=eps)  
        return y / s                              

    def _parse_group(self, a):
       
        return self.class_groups.get(int(a), [])  

    def _queue_push(self, cls, feat_cpu, soft_cpu):
      
        cls = int(cls)                            
        self.queues_feat[cls].append(feat_cpu)    
        self.queues_soft[cls].append(soft_cpu)   
        if len(self.queues_feat[cls]) > self.queue_len: 
            self.queues_feat[cls].pop(0)        
            self.queues_soft[cls].pop(0)      

    def update_queues_if_clean(self, features, noisy_cls, soft_labels):
        
        with torch.no_grad():                    
            pred = soft_labels.argmax(dim=1)      
            keep = (pred == noisy_cls)           
            if keep.sum().item() == 0:           
                return                           
            feats = features[keep].detach().to('cpu')  
            softs = soft_labels[keep].detach().to('cpu')
            cls_k = noisy_cls[keep].detach().to('cpu')  
            for i in range(feats.size(0)):      
                self._queue_push(int(cls_k[i].item()), feats[i], softs[i]) 

    def update_prototypes_ema(self, features, noisy_cls, ema_beta=0.9):
        
        with torch.no_grad():                    
            features = features.detach()          
            for c in range(self.num_class):      
                m = (noisy_cls == c)            
                if m.sum().item() == 0:          
                    continue                     
                mean_f = features[m].mean(dim=0) 
                self.prototypes[c] = (            
                    ema_beta * self.prototypes[c] 
                    + (1 - ema_beta) * mean_f     
                )

    def update_confusion_ema(self, y_true, y_pred):
        
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        conf = np.zeros((self.num_class, self.num_class), dtype=np.float32)

        for t, p in zip(y_true, y_pred):
            t = int(t); p = int(p)
            if t == p:
                continue
            conf[p, t] += 1.0

        old = self.confusion_ema.numpy()
        new = self.conf_ema * old + (1.0 - self.conf_ema) * conf
        self.confusion_ema = torch.from_numpy(new).to(self.confusion_ema.device)

    def build_class_groups(self):
        
        with torch.no_grad():
            conf_mat = self.confusion_ema.detach().cpu().numpy()

            majority = set(self.majority_classes)
            minority = set(self.minority_classes)
            middle   = set(self.middle_classes)

            cmin = len(minority)
            cmid = len(middle)
            kmin = (cmin + 1) // 2
            kmid = (cmid + 1) // 2

            for a in range(self.num_class):
               
                row = conf_mat[a].copy()
                row[a] = 0.0
                if float(row.max()) <= 0.0:
                    self.class_groups[a] = [a]
                    continue

                g = []

                if kmin > 0 and len(minority) > 0:
                    minor_cand = []
                    for m in minority:
                        m = int(m)
                        if m == a:
                            continue
                        if m in majority:
                            continue  
                        s = float(conf_mat[a, m])
                        if s > 0.0:
                            minor_cand.append((m, s))
                    minor_cand.sort(key=lambda x: -x[1])
                    g.extend([cls for cls, _ in minor_cand[:kmin]])

                if kmid > 0 and len(middle) > 0:
                    mid_cand = []
                    gset = set(g)
                    for j in middle:
                        j = int(j)
                        if j == a:
                            continue
                        if j in majority:
                            continue  
                        if j in gset:
                            continue
                        s = float(conf_mat[a, j])
                        if s > 0.0:
                            mid_cand.append((j, s))
                    mid_cand.sort(key=lambda x: -x[1])
                    g.extend([cls for cls, _ in mid_cand[:kmid]])

               
                g = [int(x) for x in g if (int(x) not in majority) and (int(x) != a)]

                self.class_groups[a] = g
    

    def _class_level_soft(self, features, anchor_cls):
        
        B = features.size(0)                          
        out = torch.zeros(B, self.num_class, device=features.device)  

        for i in range(B):                             
            a = int(anchor_cls[i].item())             
            group = self._parse_group(a)               
            cand = [a] + [int(x) for x in group]        
            cand = list(dict.fromkeys(cand))           

            proto_c = self.prototypes[cand].to(features.device)  
            sim = self._cosine_sim(features[i:i+1], proto_c).view(-1) 
            sim = torch.clamp(sim, min=0.0)             
            if sim.sum().item() <= 1e-12:             
                out[i, a] = 1.0                       
                continue                               
            w = sim / sim.sum()                      
            for j, cls_id in enumerate(cand):           
                out[i, int(cls_id)] = w[j]             
        return self._normalize_full(out)               

    def _sample_level_soft(self, features, anchor_cls):
       
        B = features.size(0)                           
        out = torch.zeros(B, self.num_class, device=features.device)  

        for i in range(B):                           
            a = int(anchor_cls[i].item())              
            group = self._parse_group(a)              
            cand_classes = [a] + [int(x) for x in group]
            cand_classes = list(dict.fromkeys(cand_classes))  

            cand_feats = []                        
            cand_softs = []                           
            for c in cand_classes:                     
                cand_feats.extend(self.queues_feat[int(c)]) 
                cand_softs.extend(self.queues_soft[int(c)])  

            if len(cand_feats) == 0:                
                continue                                

            feats = torch.stack([t.to(features.device) for t in cand_feats], dim=0)  
            softs = torch.stack([t.to(features.device) for t in cand_softs], dim=0) 

            sims = self._cosine_sim(features[i:i+1], feats).view(-1) 
            k = min(self.topk, sims.numel())            
            topv, topi = torch.topk(sims, k=k, largest=True)  

            w = torch.clamp(topv, min=0.0)             
            if w.sum().item() <= 1e-12:               
                continue                               
            w = w / w.sum()                           

            neigh_soft = softs[topi]                  
            out[i] = torch.sum(neigh_soft * w.view(-1,1), dim=0) 

        return self._normalize_full(out)             

    def build_soft_labels_clean(self, features, noisy_cls, alpha):
        
        features = features.detach()                 
        alpha = alpha.detach()                       
        B = features.size(0)                          

        y_noisy = torch.zeros(B, self.num_class, device=features.device)  
        y_noisy.scatter_(1, noisy_cls.view(-1,1), 1.0)  

        y_class = self._class_level_soft(features, noisy_cls)  
        y_samp  = self._sample_level_soft(features, noisy_cls) 

        a = alpha.view(-1,1).clamp(min=0.0, max=1.0)    
        y = (                                          
            a * y_noisy                                
            + (1.0 - a) * 0.5 * y_class                 
            + (1.0 - a) * 0.5 * y_samp               
        )

        return self._normalize_full(y)                  

    def build_soft_labels_noisy(self, features, anchor_cls):
       
        features = features.detach()                   
        y_class = self._class_level_soft(features, anchor_cls)  
        y_samp  = self._sample_level_soft(features, anchor_cls) 

        y = y_class + y_samp                        
        return self._normalize_full(y)                 

    def print_class_groups(self, epoch):
        print(f"[MSC][epoch={epoch}] class_groups:")
        for a in range(self.num_class):
            group = self.class_groups[a]
            print(f"[MSC][epoch={epoch}] g[{a}]={group}")
            
    def epoch_update_from_dataloader(self, model, dataloader, clean_index_set=None, proto_ema_beta=0.9, cache=None):

        if cache is not None:
            feats_cpu = cache["features"]  # CPU [N, D]
            y_true_cpu = cache["targets"]  # CPU [N]
            y_pred_cpu = cache["preds"]    # CPU [N]

            if clean_index_set is None:
                idx = torch.arange(y_true_cpu.numel(), dtype=torch.long)
            else:
             
                if len(clean_index_set) == 0:
                    return
                idx = torch.tensor(sorted(list(clean_index_set)), dtype=torch.long)

            if idx.numel() == 0:
                return

        
            chunk = 2048 
            for s in range(0, idx.numel(), chunk):
                j = idx[s:s+chunk]
                f = feats_cpu[j].to(self.device, dtype=torch.float32, non_blocking=True)
                t = y_true_cpu[j].to(self.device, dtype=torch.long, non_blocking=True)
                self.update_prototypes_ema(f, t, ema_beta=proto_ema_beta)

            # ---- update confusion_ema on CPU numpy ----
            y_true_np = y_true_cpu[idx].numpy().astype(np.int64)
            y_pred_np = y_pred_cpu[idx].numpy().astype(np.int64)
            self.update_confusion_ema(y_true_np, y_pred_np)
            return

        # ===== slow path: original behavior (iterate dataloader + forward) =====
        model.eval()
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for batch_idx, (inputs, _, _, targets, index, _, _) in enumerate(dataloader):
                if clean_index_set is not None:
                    idx_np = index.numpy().astype(np.int64)
                    keep = np.array([int(i) in clean_index_set for i in idx_np], dtype=np.bool_)
                    if keep.sum() == 0:
                        continue
                    inputs = inputs[keep]
                    targets = targets[keep]

                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                features, outputs = model(inputs)
                if targets.dim() > 1:
                    targets_cls = targets.argmax(dim=1)
                else:
                    targets_cls = targets
                targets_cls = targets_cls.long()

                preds = outputs.argmax(dim=1).long()

                self.update_prototypes_ema(
                    features,
                    targets_cls,
                    ema_beta=proto_ema_beta
                )

                y_true_all.append(targets_cls.detach().cpu().numpy())
                y_pred_all.append(preds.detach().cpu().numpy())

        if len(y_true_all) > 0:
            y_true_all = np.concatenate(y_true_all, axis=0)
            y_pred_all = np.concatenate(y_pred_all, axis=0)
            self.update_confusion_ema(y_true_all, y_pred_all)

    def build_soft_labels_clean_corr_only(self, features, noisy_cls):
        
        features = features.detach()
        noisy_cls = noisy_cls.detach().long()

        # ---- minority clean: no correction ----
        if len(self.minority_classes) > 0:
            minor = torch.tensor(sorted(list(self.minority_classes)),
                                device=noisy_cls.device, dtype=noisy_cls.dtype)
            # mask: [B]
            minor_mask = (noisy_cls.unsqueeze(1) == minor.unsqueeze(0)).any(dim=1)
        else:
            minor_mask = torch.zeros_like(noisy_cls, dtype=torch.bool)

        
        y_class = self._class_level_soft(features, noisy_cls)
        y_samp  = self._sample_level_soft(features, noisy_cls)
        y = self._normalize_full(y_class + y_samp)

        if minor_mask.any():
            y_onehot = torch.zeros_like(y)
            y_onehot.scatter_(1, noisy_cls.view(-1, 1), 1.0)
            y[minor_mask] = y_onehot[minor_mask]

        return y

    


class SoftGoldEvaluator:
   
    def __init__(self, num_class: int, topk=(5,)):
        self.C = int(num_class)
        self.topk = tuple(sorted(set(topk)))
        self.reset()

    def reset(self):
        C = self.C
        zL = lambda: torch.zeros(C, dtype=torch.long)
        zF = lambda: torch.zeros(C, dtype=torch.float)

        self.stat = {
            "x_total": zL(),
            "x_noisy_correct_total": zL(),
            "x_noisy_wrong_total": zL(),
            "x_soft_top1_correct_total": zL(),
            "x_soft_fix_wrong": zL(),    
            "x_soft_hurt_right": zL(),  

            "x_gold_prob_sum": zF(),
            "x_gold_prob_sum_noisy_wrong": zF(),
            "x_gold_prob_sum_noisy_right": zF(),

            "x_gold_topk_hit": {k: zL() for k in self.topk},

           
            "u_total": zL(),
            "u_pu_top1_correct": zL(),
            "u_soft_top1_correct": zL(),
            "u_soft_gain_over_pu": zL(), 
            "u_soft_drop_under_pu": zL(), 

            "u_gold_prob_sum_pu": zF(),
            "u_gold_prob_sum_soft": zF(),

            "u_gold_topk_hit": {k: zL() for k in self.topk},
        }

    @staticmethod
    def _to_hard_cls(y: torch.Tensor) -> torch.Tensor:

        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        if y.dim() > 1:
            y = y.argmax(dim=1)
        return y.long().detach().cpu()

    @staticmethod
    def _gather_gold_prob(dist: torch.Tensor, gold_cls_cpu: torch.Tensor) -> torch.Tensor:
        """
        dist: [B,C] (CPU or GPU); gold_cls_cpu: [B] CPU long
        return: [B] CPU float
        """
        if dist.is_cuda:
            dist_cpu = dist.detach().cpu()
        else:
            dist_cpu = dist.detach()
        idx = gold_cls_cpu.view(-1, 1)
        return dist_cpu.gather(1, idx).squeeze(1)

    @staticmethod
    def _topk_hit(dist: torch.Tensor, gold_cls_cpu: torch.Tensor, k: int) -> torch.Tensor:
        """
        dist: [B,C]; gold_cls_cpu: [B] CPU long
        return: [B] bool (CPU)
        """
        d = dist.detach().cpu() if dist.is_cuda else dist.detach()
        topk = torch.topk(d, k=k, dim=1).indices  # [B,k]
        return (topk == gold_cls_cpu.view(-1,1)).any(dim=1)

    def update_x(self, noisy_x: torch.Tensor, soft_x: torch.Tensor, gold_x: torch.Tensor):
        """
        noisy_x: [B] hard noisy label id (tensor)
        soft_x:  [B,C] soft distribution
        gold_x:  [B] or [B,C] gold labels (only for eval)
        """
        gold = self._to_hard_cls(gold_x)                  # CPU [B]
        noisy = noisy_x.detach().cpu().long()             # CPU [B]
        soft_top1 = soft_x.detach().cpu().argmax(dim=1)   # CPU [B]

        noisy_right = (noisy == gold)
        noisy_wrong = ~noisy_right
        soft_right = (soft_top1 == gold)

        # gold prob
        gold_prob = self._gather_gold_prob(soft_x, gold)  # CPU [B] float

        # per-class by gold
        for c in range(self.C):
            m = (gold == c)
            if m.sum().item() == 0:
                continue

            self.stat["x_total"][c] += m.sum()
            self.stat["x_noisy_correct_total"][c] += (m & noisy_right).sum()
            self.stat["x_noisy_wrong_total"][c]   += (m & noisy_wrong).sum()

            self.stat["x_soft_top1_correct_total"][c] += (m & soft_right).sum()
            self.stat["x_soft_fix_wrong"][c] += (m & noisy_wrong & soft_right).sum()
            self.stat["x_soft_hurt_right"][c] += (m & noisy_right & (~soft_right)).sum()

            self.stat["x_gold_prob_sum"][c] += gold_prob[m].sum()
            if (m & noisy_wrong).any():
                self.stat["x_gold_prob_sum_noisy_wrong"][c] += gold_prob[m & noisy_wrong].sum()
            if (m & noisy_right).any():
                self.stat["x_gold_prob_sum_noisy_right"][c] += gold_prob[m & noisy_right].sum()

            # top-k hit
            for k in self.topk:
                hit = self._topk_hit(soft_x, gold, k)  # [B] bool CPU
                self.stat["x_gold_topk_hit"][k][c] += (m & hit).sum()

    def update_u(self, pu: torch.Tensor, soft_u: torch.Tensor, gold_u: torch.Tensor):
        """
        pu:     [B,C] baseline dist (e.g., co-guess average)
        soft_u: [B,C] refined dist
        gold_u: [B] or [B,C] gold labels (only for eval)
        """
        gold = self._to_hard_cls(gold_u)                  # CPU [B]
        pu_top1 = pu.detach().cpu().argmax(dim=1)         # CPU [B]
        soft_top1 = soft_u.detach().cpu().argmax(dim=1)   # CPU [B]

        pu_right = (pu_top1 == gold)
        soft_right = (soft_top1 == gold)

        gold_prob_pu = self._gather_gold_prob(pu, gold)       # CPU [B]
        gold_prob_soft = self._gather_gold_prob(soft_u, gold) # CPU [B]

        for c in range(self.C):
            m = (gold == c)
            if m.sum().item() == 0:
                continue

            self.stat["u_total"][c] += m.sum()
            self.stat["u_pu_top1_correct"][c] += (m & pu_right).sum()
            self.stat["u_soft_top1_correct"][c] += (m & soft_right).sum()

            self.stat["u_soft_gain_over_pu"][c] += (m & (~pu_right) & soft_right).sum()
            self.stat["u_soft_drop_under_pu"][c] += (m & pu_right & (~soft_right)).sum()

            self.stat["u_gold_prob_sum_pu"][c] += gold_prob_pu[m].sum()
            self.stat["u_gold_prob_sum_soft"][c] += gold_prob_soft[m].sum()

            for k in self.topk:
                hit = self._topk_hit(soft_u, gold, k)
                self.stat["u_gold_topk_hit"][k][c] += (m & hit).sum()

    def summarize(self):
        
        rows = []
        for c in range(self.C):
            xt = int(self.stat["x_total"][c].item())
            ut = int(self.stat["u_total"][c].item())
            if xt == 0 and ut == 0:
                continue

            # X
            x_noisy_wrong = int(self.stat["x_noisy_wrong_total"][c].item())
            x_noisy_right = int(self.stat["x_noisy_correct_total"][c].item())
            x_fix = int(self.stat["x_soft_fix_wrong"][c].item())
            x_hurt = int(self.stat["x_soft_hurt_right"][c].item())

            x_soft_acc = (self.stat["x_soft_top1_correct_total"][c].item() / xt) if xt > 0 else 0.0
            x_fix_rate = (x_fix / x_noisy_wrong) if x_noisy_wrong > 0 else 0.0
            x_hurt_rate = (x_hurt / x_noisy_right) if x_noisy_right > 0 else 0.0
            x_goldP = (self.stat["x_gold_prob_sum"][c].item() / xt) if xt > 0 else 0.0

            # U
            if ut > 0:
                u_pu_acc = self.stat["u_pu_top1_correct"][c].item() / ut
                u_soft_acc = self.stat["u_soft_top1_correct"][c].item() / ut
                u_gain = int(self.stat["u_soft_gain_over_pu"][c].item())
                u_drop = int(self.stat["u_soft_drop_under_pu"][c].item())
                u_goldP_pu = self.stat["u_gold_prob_sum_pu"][c].item() / ut
                u_goldP_soft = self.stat["u_gold_prob_sum_soft"][c].item() / ut
            else:
                u_pu_acc = u_soft_acc = 0.0
                u_gain = u_drop = 0
                u_goldP_pu = u_goldP_soft = 0.0

            row = {
                "class": c,
                "x_total": xt,
                "x_soft@1_acc": float(x_soft_acc),
                "x_fix_rate(noisy_wrong->soft_right)": float(x_fix_rate),
                "x_hurt_rate(noisy_right->soft_wrong)": float(x_hurt_rate),
                "x_gold_prob_mean": float(x_goldP),

                "u_total": ut,
                "u_pu@1_acc": float(u_pu_acc),
                "u_soft@1_acc": float(u_soft_acc),
                "u_gain(soft_right&pu_wrong)": int(u_gain),
                "u_drop(soft_wrong&pu_right)": int(u_drop),
                "u_gold_prob_mean_pu": float(u_goldP_pu),
                "u_gold_prob_mean_soft": float(u_goldP_soft),
            }

            for k in self.topk:
                row[f"x_gold_top{k}_hit_rate"] = float(
                    (self.stat["x_gold_topk_hit"][k][c].item() / xt) if xt > 0 else 0.0
                )
                row[f"u_gold_top{k}_hit_rate"] = float(
                    (self.stat["u_gold_topk_hit"][k][c].item() / ut) if ut > 0 else 0.0
                )

            rows.append(row)
        return rows

    def pretty_print(self, epoch: int, max_classes: int = None):
        rows = self.summarize()
        print(f"\n[Soft-vs-Gold][Epoch {epoch}] Per-class summary (by gold class)")
        shown = 0
        for r in rows:
            c = r["class"]
            print(
                f"class={c} | "
                f"X: tot={r['x_total']} soft@1={r['x_soft@1_acc']:.3f} "
                f"fix={r['x_fix_rate(noisy_wrong->soft_right)']:.3f} "
                f"hurt={r['x_hurt_rate(noisy_right->soft_wrong)']:.3f} "
                f"goldP={r['x_gold_prob_mean']:.3f} | "
                f"U: tot={r['u_total']} pu@1={r['u_pu@1_acc']:.3f} soft@1={r['u_soft@1_acc']:.3f} "
                f"gain={r['u_gain(soft_right&pu_wrong)']} drop={r['u_drop(soft_wrong&pu_right)']} "
                f"goldP(pu/soft)={r['u_gold_prob_mean_pu']:.3f}/{r['u_gold_prob_mean_soft']:.3f}"
            )
            shown += 1
            if max_classes is not None and shown >= max_classes:
                break
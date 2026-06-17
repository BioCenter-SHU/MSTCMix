import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from imblearn.metrics import sensitivity_score, specificity_score

N_CLASSES = 9
CLASS_NAMES = [ 'ADI', 'BACK', 'LYM', 'STR', 'DEB', 'MUC', 'TUM','MUS','NORM']

def compute_AUCs(gt, pred, competition=True):
    AUROCs = []
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    indexes = range(len(CLASS_NAMES))
    for i in indexes:
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError:
            AUROCs.append(0.0)
    return AUROCs

def compute_metrics(gt, pred, competition=True, thresh=0.12):
    AUROCs, Accus, Senss, Specs = [], [], [], []
    F1s, MCCs = [], []

    TPs, FPs, FNs, TNs = [], [], [], []

    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    indexes = range(len(CLASS_NAMES))

    for i, cls in enumerate(indexes):
        y_true = gt_np[:, i].astype(np.int64)
        y_score = pred_np[:, i]
        y_pred = (y_score >= thresh).astype(np.int64)

        # --- 计算 OvR 的 TP/FP/FN/TN ---
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        TPs.append(tp); FPs.append(fp); FNs.append(fn); TNs.append(tn)

        # AUC (threshold-free)
        try:
            AUROCs.append(roc_auc_score(y_true, y_score))
        except ValueError as error:
            print(f'Error in computing AUC for {i}.\n Error msg:{error}')
            AUROCs.append(0.0)

        # Acc (thresholded)
        try:
            Accus.append(accuracy_score(y_true, y_pred))
        except ValueError as error:
            print(f'Error in computing Acc for {i}.\n Error msg:{error}')
            Accus.append(0.0)

        # Sens / Spec (thresholded)
        try:
            Senss.append(sensitivity_score(y_true, y_pred))
        except ValueError:
            print(f'Error in computing Sensitivity for {i}.')
            Senss.append(0.0)

        try:
            Specs.append(specificity_score(y_true, y_pred))
        except ValueError:
            print(f'Error in computing Specificity for {i}.')
            Specs.append(0.0)

        # F1 (thresholded)
        try:
            F1s.append(f1_score(y_true, y_pred, zero_division=0))
        except ValueError:
            print(f'Error in computing F1 for {i}.')
            F1s.append(0.0)

        # MCC (thresholded)
        try:
            MCCs.append(matthews_corrcoef(y_true, y_pred))
        except ValueError:
            print(f'Error in computing MCC for {i}.')
            MCCs.append(0.0)

    Macro_F1 = float(np.mean(F1s)) if len(F1s) > 0 else 0.0
    Macro_MCC = float(np.mean(MCCs)) if len(MCCs) > 0 else 0.0

    return AUROCs, Accus, Senss, Specs, F1s, MCCs, Macro_F1, Macro_MCC, TPs, FPs, FNs, TNs
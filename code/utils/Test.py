import torch
from torch.nn import functional as F
from utils.metrics import compute_metrics
from utils.metric_logger import MetricLogger

def epoch_metrics(net1, net2, dataLoader, class_names):
    train1 = net1.training
    net1.eval()
    train2 = net2.training
    net2.eval()

    meters = MetricLogger()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    gt_study   = {}
    pred_study = {}
    studies    = []

    with torch.no_grad():
        for it, (study, _, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, outputs1 = net1(image)
            _, outputs2 = net2(image)

            outputs = outputs1 + outputs2
            output = F.softmax(outputs, dim=1)

            for j in range(len(study)):
                if study[j] in pred_study:
                    assert torch.equal(gt_study[study[j]], label[j])
                    pred_study[study[j]] = torch.max(pred_study[study[j]], output[j])
                else:
                    gt_study[study[j]] = label[j]
                    pred_study[study[j]] = output[j]
                    studies.append(study[j])

        for s in studies:
            gt = torch.cat((gt, gt_study[s].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[s].view(1, -1)), 0)

        AUROCs, Accus, Senss, Specs, F1s, MCCs, Macro_F1, Macro_MCC, TPs, FPs, FNs, TNs = compute_metrics(
    gt, pred, competition=True)

    net1.train(train1)
    net2.train(train2)

    return AUROCs, Accus, Senss, Specs, F1s, MCCs, Macro_F1, Macro_MCC, TPs, FPs, FNs, TNs
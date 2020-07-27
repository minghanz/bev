"""For evaluating the P, R, F1, AP of predictions for keypoint-based vehicle detection, similar to that in yolov3
Only evaluate on the center estimation error, therefore inputs are center coordinates. 
"""

import numpy as np

def lin_iou(box1, box2, rbox2=None):
    """returns the ratio of center estimation error to box length. """
    # inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    dist = np.sqrt(((box1[:, None, :2] - box2[:, :2])**2).sum(axis=2))

    dist_ratio = (1 - dist / rbox2[:, 3]).clip(0)

    return dist_ratio

# def kpts_to_center(pred, conf=None):

#     if conf is None:
#         ### center estimated by all four point regardless of confidence
#         center = pred[:,:,:2].mean(axis=1) #n*2
#         conf_ctr = pred[:,:,2:3].min(axis=1) #n
#         pred_ctr = np.concatenate([center, conf_ctr], axis=1)   # n*3
#         pred_ctr = pred_ctr[(-pred_ctr[:,2]).argsort()]
#     else:
#         ### center estimated by keypoints with confidence higher than certain threshold
#         raise NotImplementedError
#         # n_inlier = (pred[:,:,2] >= conf).sum(axis=1) # n vector 
#         # idx_4 = (n_inlier == 4).nonzero().view(-1)
#         # idx_3 = (n_inlier == 3).nonzero().view(-1)
#         # idx_2 = (n_inlier == 2).nonzero().view(-1)
    
#     return pred_ctr

def eval_single_image_prep(pred_ctr, tgt_ctr, rbox_target):
    """pred: n*4*3(x,y,conf)
       target: m*4*3(x,y,1), 
       rbox_target: m*5, (we need the dimension)"""
    ### find the best target for each prediction
    ### determine each prediction is TP or FP

    iouv = np.linspace(0.5, 0.95, 10)
    iouv = iouv[[0]]
    niou = len(iouv)

    # if pred_conf is not None:
    #     pred[:,:, 2] = pred_conf[:,None]
    # pred_ctr = kpts_to_center(pred)
    # tgt_ctr = kpts_to_center(target) 



    # ti = np.array(range(target.shape[0]))
    # pi = np.array(range(pred.shape[0]))
    correct = np.zeros((pred_ctr.shape[0], niou), dtype=bool)
    detected = []
    nl = tgt_ctr.shape[0]

    dist_ratio = lin_iou(pred_ctr, tgt_ctr, rbox_target)    # n*m
    ious = dist_ratio.max(axis=1)                           # n
    i = dist_ratio.argmax(axis=1)

    js = np.argwhere(ious > iouv[0]).reshape(-1)
    for j in js:
        ti_d = i[j] #ti[i[j]]
        if ti_d not in detected:
            detected.append(ti_d)
            correct[j] = ious[j] >= iouv # correct[pi[j]]
            if len(detected) == nl:
                break
    
    return correct, nl

def eval_single_image(pred_ctr, tgt_ctr, rbox_target, pred_conf):
    """pred: n*4*3(x,y,conf)
       target: m*4*3(x,y,1), 
       rbox_target: m*5, (we need the dimension)"""
    correct, nl = eval_single_image_prep(pred_ctr, tgt_ctr, rbox_target)
    
    p, r, ap, f1 = calc_prapf1(correct, pred_conf, nl )
    if niou > 1:
        mp_iou, mr_iou, map_iou, mf1_iou = p.mean(), r.mean(), ap.mean(), f1.mean()  # [P, R, AP@0.5:0.95, AP@0.5]
        ### take average over iou_thresholds
    else:
        mp_iou, mr_iou, map_iou, mf1_iou = p, r, ap, f1

    p0, r0, ap0, f10 = p[0], r[0], ap[0], f1[0]

    return np.array([mp_iou, mr_iou, map_iou, mf1_iou]), np.array([p0, r0, ap0, f10])


def calc_prapf1(correct, conf, n_target):
    """
    correct: npred*niou, conf: n, n_target: m
    """
    pr_conf = 0.5

    s = [correct.shape[1]]  # number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)

    tp = correct.cumsum(0)

    fp = (1-correct).cumsum(0)

    recall = tp / (n_target+1e-16)
    precision = tp / (tp+fp)

    for j in range(correct.shape[1]):
        ap[j] = compute_ap(recall[:, j], precision[:, j])

        r[j] = np.interp(-pr_conf, -conf, recall[:, j])
        p[j] = np.interp(-pr_conf, -conf, precision[:, j])

    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

    # confv = np.linspace(0.1, 1, 10)
    # for i in range(len(confv)):
    #     pred_ctr = kpts_to_center(pred, confv[i])
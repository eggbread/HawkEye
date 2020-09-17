from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(cfg, use_cuda):
    return DeepSort(cfg['DEEPSORT']['REID_CKPT'],
                max_dist=float(cfg['DEEPSORT']['MAX_DIST']), min_confidence=float(cfg['DEEPSORT']['MIN_CONFIDENCE']),
                nms_max_overlap=float(cfg['DEEPSORT']['NMS_MAX_OVERLAP']), max_iou_distance=float(cfg['DEEPSORT']['MAX_IOU_DISTANCE']),
                max_age=int(cfg['DEEPSORT']['MAX_AGE']), n_init=int(cfg['DEEPSORT']['N_INIT']), nn_budget=int(cfg['DEEPSORT']['NN_BUDGET']), use_cuda=use_cuda)
    










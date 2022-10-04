import time
import torch

import sys
import os

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
import cudaprofile
import params
from torch_wrapper import benchmark_wrapper


from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)

import mmdet.models.dense_heads.anchor_head as X

dev = torch.device(params.devname)

def loss(self,
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore=None):

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    assert len(featmap_sizes) == self.prior_generator.num_levels

    device = cls_scores[0].device

    anchor_list, valid_flag_list = self.get_anchors(
        featmap_sizes, img_metas, device=device)
    label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
    cls_reg_targets = self.get_targets(
        anchor_list,
        valid_flag_list,
        gt_bboxes,
        img_metas,
        gt_bboxes_ignore_list=gt_bboxes_ignore,
        gt_labels_list=gt_labels,
        label_channels=label_channels)
    if cls_reg_targets is None:
        return None
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        num_total_pos, num_total_neg) = cls_reg_targets
    num_total_samples = (
        num_total_pos + num_total_neg if self.sampling else num_total_pos)

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    concat_anchor_list = []
    for i in range(len(anchor_list)):
        concat_anchor_list.append(torch.cat(anchor_list[i]))
    all_anchor_list = images_to_levels(concat_anchor_list,
                                        num_level_anchors)

    losses_cls, losses_bbox = multi_apply(
        self.loss_single,
        cls_scores,
        bbox_preds,
        all_anchor_list,
        labels_list,
        label_weights_list,
        bbox_targets_list,
        bbox_weights_list,
        num_total_samples=num_total_samples)
    return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

print('Monkey patching AnchorHead.loss...')
X.AnchorHead.loss = loss

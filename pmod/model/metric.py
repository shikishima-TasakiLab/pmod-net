from pmod.model.utils import quat_inv, quat_mul
from pmod.model.constant import METRIC_SUM_OUTPUT, Q_IDX
from typing import Dict, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LossNULL(nn.Module):
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        super(LossNULL, self).__init__()
        self.device = device

    def to(self, device: torch.device) -> None:
        self.device = device

    def forward(self, *args):
        return torch.tensor(0, device=self.device)


class InRange(nn.Module):
    def __init__(self, range_up: float = 1.0, num_classes: int = 0) -> None:
        super(InRange, self).__init__()

        self.range_up: Tensor
        self.register_buffer('range_up', torch.tensor(range_up))

        self.num_classes: Tensor
        self.register_buffer('num_classes', torch.tensor(num_classes))

    def in_range(self, gt: Tensor, seg_gt: Tensor = None) -> Tuple[Tensor, Tensor]:
        in_range = (0. < gt) & (gt < self.range_up)
        if self.num_classes > 0:
            in_range = in_range.squeeze(dim=1)
            in_range_oh: Tensor = F.one_hot(
                seg_gt, num_classes=self.num_classes)
            in_range_oh[~in_range] = False
            return in_range_oh.bool(), torch.sum(in_range_oh, dim=(1, 2))
        else:
            return in_range, torch.sum(in_range, dim=(1, 2, 3))


class LossL1(InRange):
    def __init__(self, range_up: float = 1.0) -> None:
        super(LossL1, self).__init__(range_up)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        zeros = torch.zeros_like(gt)
        in_range, cnts = self.in_range(gt)
        l1 = torch.where(in_range, torch.abs(pred - gt), zeros)
        return torch.sum(l1) / torch.sum(cnts)


class MetricInRange(nn.Module):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricInRange, self).__init__()

        thresholds.sort()
        self.thresholds: Tensor
        self.register_buffer('thresholds', torch.tensor(thresholds))

        self.num_classes: Tensor
        self.register_buffer('num_classes', torch.tensor(num_classes))

    def in_range(self, gt: Tensor, seg_gt: Tensor = None) -> Tuple[Tensor, Tensor]:
        _range_list: List[Tensor] = []
        _prev_th: float = 0.0
        if self.num_classes > 0:
            _one_hot: Tensor = F.one_hot(seg_gt, num_classes=self.num_classes)
        for th_idx in range(self.thresholds.shape[0]):
            _th: Tensor = self.thresholds[th_idx].to(gt.device)
            if _th <= _prev_th:
                continue
            _in_range: Tensor = (_prev_th < gt) & (gt <= _th)  # NCHW
            if self.num_classes > 0:
                _in_range: Tensor = torch.squeeze(_in_range, dim=1)  # NHW
                _in_range_oh: Tensor = _one_hot.clone()  # NHWC
                _in_range_oh[~_in_range] = False
                _range_list.append(_in_range_oh.bool().permute(
                    0, 3, 1, 2).unsqueeze(dim=1))  # NTCHW
            else:
                _range_list.append(_in_range.unsqueeze(dim=1))  # NTCHW
            _prev_th = _th

        range_cat: Tensor = torch.cat(_range_list, dim=1)  # NTCHW
        range_cnt: Tensor = range_cat.sum(
            dim=(3, 4)) if self.num_classes > 0 else range_cat.sum(dim=(2, 3, 4))
        return range_cat, range_cnt


class MetricBatchSumAPE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumAPE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        ape = torch.abs(pred - gt) / gt  # NCHW

        ape = ape.unsqueeze(dim=1).expand_as(in_range)  # NTCHW
        zeros = torch.zeros_like(ape)
        ape = torch.where(in_range, ape, zeros)

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(ape.sum(dim=(3, 4)), cnts)  # NTC
        else:
            return METRIC_SUM_OUTPUT(ape.sum(dim=(2, 3, 4)), cnts)  # NT


class MetricSumAPE(MetricBatchSumAPE):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricSumAPE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        ape = super().forward(pred, gt, seg_gt)
        return METRIC_SUM_OUTPUT(torch.sum(ape.sum, dim=0), torch.sum(ape.count, dim=0))


class MetricIntersection(nn.Module):
    def __init__(self):
        super(MetricIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        pred = torch.argmax(pred, dim=1)
        pred = F.one_hot(pred, num_classes=num_classes)
        gt = F.one_hot(gt, num_classes=num_classes)

        intersection = torch.logical_and(pred, gt)

        return torch.sum(intersection, dim=(1, 2))


class MetricSumIntersection(MetricIntersection):
    def __init__(self):
        super(MetricSumIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt), dim=0)


class MetricUnion(nn.Module):
    def __init__(self, smooth: float = 1e-10):
        super(MetricUnion, self).__init__()
        self.smooth = torch.tensor(smooth)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        pred = torch.argmax(pred, dim=1)
        pred = F.one_hot(pred, num_classes=num_classes)
        gt = F.one_hot(gt, num_classes=num_classes)

        union = torch.logical_or(pred, gt)

        return torch.sum(union, (1, 2)) + self.smooth


class MetricSumUnion(MetricUnion):
    def __init__(self, smooth: float = 1e-10):
        super().__init__(smooth)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt) - self.smooth, dim=0) + self.smooth


class MetricBatchSumAE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumAE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        ae = torch.abs(pred - gt)  # NCHW

        ae = ae.unsqueeze(dim=1).expand_as(in_range)  # NTCHW
        zeros = torch.zeros_like(ae)
        ae = torch.where(in_range, ae, zeros)

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(ae.sum(dim=(3, 4)), cnts)  # NTC
        else:
            return METRIC_SUM_OUTPUT(ae.sum(dim=(2, 3, 4)), cnts)  # NT


class MetricSumAE(MetricBatchSumAE):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricSumAE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        ae = super().forward(pred, gt, seg_gt)
        return METRIC_SUM_OUTPUT(torch.sum(ae.sum, dim=0), torch.sum(ae.count, dim=0))


class MetricBatchSumSE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumSE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        se = torch.pow(pred - gt, 2)  # NCHW

        se = se.unsqueeze(dim=1).expand_as(in_range)  # NTCHW
        zeros = torch.zeros_like(se)
        se = torch.where(in_range, se, zeros)

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(se.sum(dim=(3, 4)), cnts)  # NTC
        else:
            return METRIC_SUM_OUTPUT(se.sum(dim=(2, 3, 4)), cnts)  # NT


class MetricSumSE(MetricBatchSumSE):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricSumSE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        se = super().forward(pred, gt, seg_gt)
        return METRIC_SUM_OUTPUT(torch.sum(se.sum, dim=0), torch.sum(se.count, dim=0))

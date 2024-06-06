from dataclasses import dataclass
from typing import Dict, Optional

from elias.config import Config
from visage.evaluator.paired_image_evaluator import PairedImageMetrics
from visage.evaluator.paired_video_evaluator import PairedVideoMetric


@dataclass
class PerSequenceMetric(Config):
    paired_image_metrics: PairedImageMetrics
    paired_video_metric: Optional[PairedVideoMetric] = None

    def __add__(self, other) -> 'PerSequenceMetric':
        return PerSequenceMetric(
            paired_image_metrics=self.paired_image_metrics + other.paired_image_metrics,
            paired_video_metric=self.paired_video_metric + other.paired_video_metric
        )

    def __radd__(self, other) -> 'PerSequenceMetric':
        if other == 0:
            return self

        return self + other

    def __truediv__(self, scalar: float) -> 'PerSequenceMetric':
        return PerSequenceMetric(
            paired_image_metrics=self.paired_image_metrics / scalar,
            paired_video_metric=self.paired_video_metric / scalar)


@dataclass
class DiffusionAvatarsEvaluationResult(Config):
    per_sequence_metrics: Dict[str, Dict[str, Dict[str, PerSequenceMetric]]]  # {p_id: {sequence: {serial: metric}}}
    average_per_sequence_metric: Optional[PerSequenceMetric] = None
    masked_per_sequence_metrics: Optional[Dict[str, Dict[str, Dict[str, PerSequenceMetric]]]] = None  # {p_id: {sequence: {serial: metric}}}
    masked_average_per_sequence_metric: Optional[PerSequenceMetric] = None

    def get_average_metrics(self, masked: bool = False) -> Dict[str, float]:
        average_metrics = dict()
        if masked:
            if self.masked_average_per_sequence_metric.paired_video_metric is not None:
                average_metrics["jod"] = self.masked_average_per_sequence_metric.paired_video_metric.jod
            average_metrics["psnr"] = self.masked_average_per_sequence_metric.paired_image_metrics.psnr
            average_metrics["ssim"] = self.masked_average_per_sequence_metric.paired_image_metrics.ssim
            average_metrics["multi_scale_ssim"] = self.masked_average_per_sequence_metric.paired_image_metrics.multi_scale_ssim
            average_metrics["lpips"] = self.masked_average_per_sequence_metric.paired_image_metrics.lpips
            average_metrics["mse"] = self.masked_average_per_sequence_metric.paired_image_metrics.mse
            average_metrics["akd"] = self.masked_average_per_sequence_metric.paired_image_metrics.akd
            average_metrics["csim"] = self.masked_average_per_sequence_metric.paired_image_metrics.csim
            average_metrics["aed"] = self.masked_average_per_sequence_metric.paired_image_metrics.aed
            average_metrics["apd"] = self.masked_average_per_sequence_metric.paired_image_metrics.apd
        else:
            if self.average_per_sequence_metric.paired_video_metric is not None:
                average_metrics["jod"] = self.average_per_sequence_metric.paired_video_metric.jod
            average_metrics["psnr"] = self.average_per_sequence_metric.paired_image_metrics.psnr
            average_metrics["ssim"] = self.average_per_sequence_metric.paired_image_metrics.ssim
            average_metrics["multi_scale_ssim"] = self.average_per_sequence_metric.paired_image_metrics.multi_scale_ssim
            average_metrics["lpips"] = self.average_per_sequence_metric.paired_image_metrics.lpips
            average_metrics["mse"] = self.average_per_sequence_metric.paired_image_metrics.mse
            average_metrics["akd"] = self.average_per_sequence_metric.paired_image_metrics.akd
            average_metrics["csim"] = self.average_per_sequence_metric.paired_image_metrics.csim
            average_metrics["aed"] = self.average_per_sequence_metric.paired_image_metrics.aed
            average_metrics["apd"] = self.average_per_sequence_metric.paired_image_metrics.apd

        return average_metrics

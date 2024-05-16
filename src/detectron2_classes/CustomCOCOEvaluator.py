# Modify coco evaluater to use CustomCOCOEval
# source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py

import os
import json
import copy
import itertools
import numpy as np
from fvcore.common.file_io import PathManager
from detectron2.evaluation import COCOEvaluator
from .CustomCOCOEval import CustomCOCOEval
        
class CustomCOCOEvaluator(COCOEvaluator):
    
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, min_iou=0.05):
        super().__init__(dataset_name, cfg, distributed=distributed)
        self.min_iou = min_iou
        if self.min_iou:
            print(f"Using custom min iou: {self.min_iou}")
    
    def _evaluate_predictions_on_coco(self, coco_gt, coco_results, iou_type, 
                                  kpt_oks_sigmas=None, min_iou=0.05):
        """
        Evaluate the coco results using COCOEval API.
        """
        assert len(coco_results) > 0

        # configure coco_eval
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = CustomCOCOEval(cocoGt=coco_gt, cocoDt=coco_dt, iouType=iou_type)
        coco_eval.params.maxDets = [512]
        new_iou_thresh = np.linspace(min_iou, 0.95, int(np.round((0.95 - min_iou) / .05)) + 1, endpoint=True)
        coco_eval.params.iouThrs = new_iou_thresh

        # calculate evaluation score
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval
    
    def _eval_predictions(self, tasks, predictions, min_iou=0.05):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        
        # Only "bbox" in our case
        for task in sorted(tasks):
            coco_eval = (
                self._evaluate_predictions_on_coco(
                    self._coco_api, 
                    coco_results, 
                    task, 
                    kpt_oks_sigmas=self._kpt_oks_sigmas, 
                    min_iou=self.min_iou
                )
                if len(coco_results) > 0
                else None  
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )

            self._results[task] = {'precision': coco_eval.eval['precision'],
                                   'params': coco_eval.eval['params'],
                                   'res': res,
                                   'scores': coco_eval.eval['scores'],
                                   'coco_eval': coco_eval.eval
                                  }   
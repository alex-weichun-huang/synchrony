# Modify trainer to include custom hook
# source: https://github.com/facebookresearch/detectron2/blob/3c7bb714795edc7a96c9a1a6dd83663ecd293e36/detectron2/engine/defaults.py#LL321C33-L321C34

import os 
import wandb
from detectron2.engine import DefaultTrainer
from detectron2.engine import hooks
import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader
from detectron2.data import build_detection_test_loader
from .CustomCOCOEvaluator import CustomCOCOEvaluator
from .CustomDatasetMapper import CustomDatasetMapper
from .CustomLossEvalHook import CustomLossEvalHook


class CustomTrainer(DefaultTrainer):
    
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")   
        return CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)  
    
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader` 
            but now with mapper.
        Overwrite it if you'd like a different data loader.
        """
        mapper = CustomDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
    
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = CustomDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    
    def __init__(self, cfg, use_wandb=True):
        """
        Args:
            cfg (CfgNode):
        """
        self.use_wandb = use_wandb
        super().__init__(cfg)

    def build_hooks(self,):
        """
        Modified from default trainer build_hooks.
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        
        Modifyied to calculate validation loss and change learning 
        rate on validation loss.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        
        hooks_list = [
            # HOOK 1: timer
            hooks.IterationTimer(),
            
            # HOOK 2: learning rate scheduler
            hooks.LRScheduler(),
        ]


        # HOOK 3: save checkpoints
        if cfg.SOLVER.CHECKPOINT_PERIOD != 0:
            if comm.is_main_process():
                hooks_list.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))


        # HOOK 4: log APs to wandb if use_wandb is True, else only save the results
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model) 
            if self.use_wandb:
                logs = self._last_eval_results["bbox"]["res"]
                wandb.log({
                    'AP': logs['AP'], 
                    'AP50': logs['AP50'], 
                    'AP75': logs['AP75'], 
                    'APs': logs['APs'], 
                    'APm': logs['APm'], 
                    'APl': logs['APl']
                }, step= self.iter)
                print("Logs sent to wandb\n")
            return self._last_eval_results["bbox"]["res"]
        hooks_list.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        
        
        # HOOK 5: to caclulate the validation loss
        checkpointer = self.checkpointer
        mapper = CustomDatasetMapper(cfg, is_train=False, calc_val_loss=True)
        loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], mapper)
        hooks_list.append(CustomLossEvalHook(cfg.TEST.EVAL_PERIOD, self.model, loader, checkpointer))
        
        
        # HOOK 6: write events to EventStorage
        if comm.is_main_process():
            hooks_list.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        
        
        return hooks_list
    
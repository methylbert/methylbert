#!/usr/bin/env python3
import fire
import numpy as np
import pandas as pd
import json
import warnings
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchmetrics

import pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from light.core.logger import ConsoleLogger
from light.core.random import worker_init_fn_seed


def hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)



class MethyDataModule(pl.LightningDataModule):
    """
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, df_path, data_args={},
                 task='pretrain',
                 col_group='dataset', batch_size=32, num_workers=0, pin_memory=True, shuffle=True,
                 ):
        super().__init__()
        self.df_path = df_path
        self.data_args = data_args

        self.task = task

        self.col_group = col_group
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        filename = Path(self.df_path).name
        if str(self.df_path).endswith('.csv'):
            df = pd.read_csv(self.df_path, low_memory=False)
            if self.col_group in df: # col_group can be used to divide the dataset in advance
                df_train = df[df[self.col_group].isin(['train'])]
                df_valid = df[df[self.col_group].isin(['valid', 'dev'])]
                df_test = df[df[self.col_group].isin(['valid', 'dev', 'test', 'other'])]
            else:
                warnings.warn('Not have col_group')
                df_train = df
                df_valid = df
                df_test = df
        else:
            df_train = self.df_path
            df_valid = self.df_path
            df_test = self.df_path

        if self.task == 'pretrain':
            from methy.data.methy_data import MethyPretrainDataset
            DatasetCls = MethyPretrainDataset
        elif self.task == 'diag':
            from methy.data.methy_data import MethyDiagDataset
            DatasetCls = MethyDiagDataset
        else:
            raise NotImplementedError(f'Not implemented task: {self.task}')
        self.ds_train = DatasetCls(df_train, **self.data_args)
        self.ds_valid = DatasetCls(df_valid, **self.data_args)
        self.ds_test = DatasetCls(df_test, **self.data_args)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=self.shuffle, worker_init_fn=worker_init_fn_seed)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False, worker_init_fn=worker_init_fn_seed)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False, worker_init_fn=worker_init_fn_seed)

    def teardown(self, stage=None):
        pass


def get_model(task='pretrain', model_name='pretrain', num_classes=1, model_path=None, model_args={}):
    if model_path is None:  # create a new model
        if task == 'pretrain':
            from methy.model.methy_performer import PerformerLM
            model_args_base = dict(
                num_classes=num_classes,
                dim=64,
                depth=3,
                heads=4,
                dim_head=16,
            )
            model_args_base.update(model_args)
            model = PerformerLM(**model_args_base)
        elif task in ['diag']:
            from methy.model.methy_performer import PerformerDiag
            model_args_base = dict(
                num_classes=num_classes,
            )
            model_args_base.update(model_args)
            model = PerformerDiag(**model_args_base)
        else:
            raise NotImplementedError(f'The task is not implemented: {task}')
    else:
        model = torch.load(model_path)
    return model


class PretrainModelModule(pl.LightningModule):
    def __init__(self, model_name, model, num_classes=None, output_dir=None,
                 lr=0.001, model_args=None, data_args=None,
                 ignore_index=-100,
                 ):
        super().__init__()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.lr = lr
        self.num_classes = num_classes if num_classes is not None else model.num_classes
        self.model_name = model_name
        self.model_args = model_args
        self.model = model
        self.IGNORE_INDEX = ignore_index
        self.data_args = data_args

        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_loss = None

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        out = self.model(*x)
        return out

    def on_train_start(self):
        log_hyperparams = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "data_args": self.data_args,
            "lr": self.lr,
        }
        log_hyperparams.update(self.model_args)
        self.logger.log_hyperparams(log_hyperparams)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(0, 1), ignore_index=self.IGNORE_INDEX)
        self.train_loss = loss.detach()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(0, 1), ignore_index=self.IGNORE_INDEX)

        if self.train_loss is None:
            self.train_loss = 0
        self.log("train_loss", self.train_loss, prog_bar=False, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        return {}

    def test_epoch_end(self, outputs):
        pass


class DiagModelModule(pl.LightningModule):
    def __init__(self, model_name, model, num_classes=None, output_dir=None,
                 lr=0.001, model_args=None, data_args=None,
                 ignore_index=-100,
                 ):
        super().__init__()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.lr = lr
        self.num_classes = num_classes if num_classes is not None else model.num_classes
        self.model_name = model_name
        self.model_args = model_args
        self.model = model
        self.IGNORE_INDEX = ignore_index
        self.data_args = data_args

        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_loss = None
        metrics = {
            'cancer': nn.ModuleDict(dict(
                auc=torchmetrics.classification.AUROC(),
                acc=torchmetrics.classification.Accuracy(),
            )),
            'loss': nn.ModuleDict(dict(
                loss_cancer=torchmetrics.aggregation.MeanMetric(),
            )),
        }
        self.metrics = nn.ModuleDict(metrics)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        out = self.model(*x)
        return out

    def on_train_start(self):
        log_hyperparams = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "data_args": self.data_args,
            "lr": self.lr,
        }
        log_hyperparams.update(self.model_args)
        self.logger.log_hyperparams(log_hyperparams)

    def get_loss(self, preds, y):
        preds_cancer = preds[..., 0]
        y_cancer = y[..., 0]
        mask_cancer = (y_cancer != -100)
        loss_cancer = F.binary_cross_entropy_with_logits(preds_cancer, y_cancer.float(),reduction='none')
        loss_cancer = (loss_cancer*mask_cancer).sum()/(mask_cancer.sum().clip(min=1))

        if torch.isnan(loss_cancer):
            loss_cancer = 0.

        loss = loss_cancer
        losses = {
            'loss_cancer': loss_cancer,
            'loss': loss,
        }
        return losses

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        losses = self.get_loss(logits, y)
        loss = losses['loss']
        self.train_loss = loss.detach()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        losses = self.get_loss(logits, y)
        loss = losses['loss']
        loss_cancer = losses['loss_cancer']

        logits_cancer = logits[:, 0]
        y_cancer = y[..., 0]
        preds_cancer = (logits_cancer > 0.5).long()

        mask = y_cancer != -100
        if mask.sum() >= 1:
            logits_cancer = logits_cancer[mask]
            preds_cancer = preds_cancer[mask]
            y_cancer = y_cancer[mask]
        else:
            logits_cancer = logits_cancer[:1]
            preds_cancer = preds_cancer[:1]
            y_cancer = y_cancer[:1]
            y_cancer[:1] = 0

        if self.train_loss is None:
            self.train_loss = 0
        self.log("train_loss", self.train_loss, prog_bar=False, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)

        # cancer eval
        for metrics_name in ['auc', 'acc']:
            metrics = self.metrics['cancer'][metrics_name]
            if metrics_name in ['auc']:
                metrics.update(logits_cancer.flatten(), y_cancer.flatten())
            else:
                metrics.update(preds_cancer.flatten(), y_cancer.flatten())
            metrics_attr_name = f"val_cancer_{metrics_name}"
            self.log(metrics_attr_name, metrics, prog_bar=False, sync_dist=True,
                     metric_attribute=metrics_attr_name)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        return {'pred': logits, 'true': y}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['pred'] for x in outputs], dim=0)
        trues = torch.cat([x['true'] for x in outputs], dim=0)
        trues = trues.to('cpu')  # (B, 2)
        preds = preds.to('cpu')  # (B, 1+k)
        preds_cancer = preds[:,0].sigmoid()
        probs_loc = torch.softmax(preds[:,1:],dim=-1)

        global_rank = self.global_rank
        indices = np.arange(len(trues)).astype('int')
        output_path = self.output_dir / f'test_pred.{global_rank}.csv'
        result = {
            'sample_id': indices,
            'true_cancer': trues[:,0],
            'true_loc': trues[:,1],
            'pred_cancer': preds_cancer,
        }
        df_pred = pd.DataFrame(result)
        for i in range(probs_loc.size(-1)):
            df_pred[f'prob_loc_{i}'] = probs_loc[:, i]

        df_pred.to_csv(output_path, index=False)


class MethyLightRunner(object):
    def __init__(self, valid_ratio=0.2, gpus=0,
                 batch_size=8, num_workers=0,
                 pin_memory=True,
                 not_find_unused=False,
                 static_graph=False,
                 fp16=False, accelerator='cpu',
                 monitor_name='val_loss', monitor_mode='min',
                 ):
        """
        device: str or None (default=None)
        """
        super(MethyLightRunner, self).__init__()
        self._gpus = gpus
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._valid_ratio = valid_ratio
        self._pin_memory = pin_memory
        self._not_find_unused = not_find_unused
        self._static_graph = static_graph
        self._fp16 = fp16
        if self._gpus == 0:
            self._gpus = None
        else:
            accelerator = 'cuda'
        self._accelerator = accelerator
        self._monitor_name = monitor_name
        self._monitor_mode = monitor_mode

        if self._accelerator == 'mps':
            self._strategy = None
        else:
            self._strategy = DDPStrategy(static_graph=self._static_graph)

    def train(self, task, df_path, data_root, output_dir, *, col_group='dataset', debug=False,
              model_name='seq',
              shuffle=True, lr=0.001, n_epoch=10, patience=10,
              model_args={}, data_args={}, module_args={}, data_config=None,
              model_checkpoint=None,
              ):
        """
        Train

        Args:
        --------------------
        df_path: str
            data table file path
        data_root: str
            data root directory path
        output_dir: str
            output directory path
        """

        output_dir = Path(output_dir)

        if data_config is not None:
            data_args_default = json.loads(Path(data_config).read_text())
            data_args_default.update(data_args)
            data_args = data_args_default
        data_args.update({'data_root': data_root})

        # data module
        data_module = MethyDataModule(df_path, task=task, data_args=data_args, col_group=col_group,
                                      batch_size=self._batch_size, num_workers=self._num_workers,
                                      pin_memory=self._pin_memory, shuffle=shuffle,
                                      )
        data_module.setup()
        num_classes = data_module.ds_train.num_classes

        # model module
        model = get_model(task=task, model_name=model_name, num_classes=num_classes,
                          model_args=model_args)
        module_args = dict(model_name=model_name, model=model, lr=lr,
                           output_dir=output_dir, model_args=model_args, data_args=data_args)
        if task == 'pretrain':
            ModelModule = PretrainModelModule
        elif task == 'diag':
            ModelModule = DiagModelModule
        else:
            raise NotImplementedError(f'Not implemented task: {task}')
        if model_checkpoint is None:
            model_module = ModelModule(**module_args)
        else: # load from checkpoint
            model_module = ModelModule.load_from_checkpoint(model_checkpoint,**module_args)

        # trainer
        log_dir = output_dir / 'log'
        logger_csv = CSVLogger(str(log_dir))
        version_dir = Path(logger_csv.log_dir)

        if self._not_find_unused:
            pytorch_lightning.plugins.DDPPlugin(find_unused_parameters=False)
        trainer_args = {}
        if self._fp16:
            trainer_args.update(dict(amp_backend="native", precision=16))

        trainer = Trainer(
            accelerator=self._accelerator,
            # gpus=self._gpus,
            devices=self._gpus,
            max_epochs=n_epoch,
            logger=[
                ConsoleLogger(),
                logger_csv,
            ],
            callbacks=[
                EarlyStopping(monitor=self._monitor_name, mode=self._monitor_mode, patience=patience),
                ModelCheckpoint(dirpath=(version_dir / 'checkpoint'), filename='{epoch}-{val_loss:.3f}',
                                monitor=self._monitor_name, mode=self._monitor_mode, save_last=True),
                TQDMProgressBar(refresh_rate=1),
            ],
            # strategy='ddp',
            # strategy=DDPPlugin(static_graph=self._static_graph),
            strategy=self._strategy,
            **trainer_args,
        )
        trainer.fit(model_module, datamodule=data_module)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        model_module = model_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, **module_args)

        # Test
        dl_test = data_module.test_dataloader()
        test_eval = trainer.validate(model_module, dataloaders=dl_test)
        if trainer.global_rank == 0:
            Path(output_dir / 'test_eval.json').write_text(json.dumps(test_eval, indent=2))
            dl_test.dataset.df.to_csv(output_dir / 'test_data.csv', index=False)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        trainer.test(model_module, dataloaders=dl_test)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if trainer.global_rank == 0:
            paths = sorted((output_dir / 'pred').glob('*.csv'))
            dfs_pred = []
            for path in paths:
                global_rank = int(path.name.split('.')[1])
                df_pred = pd.read_csv(path)
                df_pred['sample_id'] = df_pred['sample_id'] * trainer.world_size + global_rank
                dfs_pred += [df_pred]
            if len(dfs_pred) > 0:
                df_pred = pd.concat(dfs_pred).sort_values('sample_id')
                df_pred.to_csv(output_dir / 'test_pred.csv', index=False)

        # Export model
        if trainer.global_rank == 0:
            model = model_module.model
            (output_dir / 'model_data.json').write_text(json.dumps(data_args, indent=2))
            torch.save(model.state_dict(), str(output_dir / 'state_dict.zip'))
            torch.save(model, str(output_dir / 'model.pt'))
            torch.save(model.state_dict(), str(version_dir / 'state_dict.zip'))
            torch.save(model, str(version_dir / 'model.pt'))

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def test(self, task, df_path, data_root, output_dir, model_path, col_group='dataset', debug=False,
             data_args={},
             col_id=None, split_i=None, split_n=None, split_salt='',
             ):
        """
        测试


        Args:
        --------------------
        df_path: str
            data table file path
        data_root: str
            data root directory path
        model_path: str
            model file path
        output_dir: str
            output directory path
        """

        model_data_path = Path(model_path).parent / 'model_data.json'
        output_dir = Path(output_dir)

        # model module
        model = get_model(model_path=model_path)
        module_args = dict(model_name=model_path, model=model,
                           output_dir=output_dir, model_args={}, data_args={})
        if task == 'pretrain':
            ModelModuleCls = PretrainModelModule
        elif task == 'diag':
            ModelModuleCls = DiagModelModule
        model_module = ModelModuleCls(**module_args)

        if model_data_path.exists():
            data_args_pre = json.loads(model_data_path.read_text())
        else:
            data_args_pre = {}
        print(data_args_pre)
        # data_args_pre = data_args_log.get('data_args', {})
        data_args_pre.update({'data_root': data_root})
        data_args_pre.update(data_args) # update data_args

        # data module
        data_module = MethyDataModule(df_path, task=task, data_args=data_args_pre,
                                      col_group=col_group, batch_size=self._batch_size,
                                      num_workers=self._num_workers,
                                      pin_memory=self._pin_memory, shuffle=False,
                                      )

        # trainer
        log_dir = output_dir / 'log'
        logger_csv = CSVLogger(str(log_dir))
        trainer = Trainer(
            accelerator=self._accelerator,
            # gpus=self._gpus,
            devices=self._gpus,
            max_epochs=0,
            logger=[
                logger_csv,
            ],
            callbacks=[
                TQDMProgressBar(refresh_rate=1),
            ],
        )

        # Test
        data_module.setup()
        dl_test = data_module.test_dataloader()
        dl_test.dataset.df.to_csv(output_dir / 'test_data.csv', index=False)
        trainer.test(model_module, dataloaders=dl_test)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if trainer.global_rank == 0:
            paths = sorted((output_dir / 'pred').glob('*.csv'))
            dfs_pred = []
            for path in paths:
                global_rank = int(path.name.split('.')[1])
                df_pred = pd.read_csv(path)
                df_pred['sample_id'] = df_pred['sample_id'] * len(paths) + global_rank
                dfs_pred += [df_pred]
            if len(dfs_pred) > 0:
                df_pred = pd.concat(dfs_pred).sort_values('sample_id')
                df_pred.to_csv(output_dir / 'test_pred.csv', index=False)



if __name__ == '__main__':
    fire.Fire(MethyLightRunner)

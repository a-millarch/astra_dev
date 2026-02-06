import os
import warnings
import copy

import pandas as pd 
import numpy as np 
import pickle 

from omegaconf import OmegaConf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from tsai.data.core import get_ts_dls
from tsai.data.preprocessing import TSStandardize, TSNormalize, TSMultiLabelClassification
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.validation import get_splits
from tsai.data.preparation import df2xy
from tsai.all import TensorMultiCategory
from tsai.all import computer_setup, TSTPlus, count_parameters, TSTabFusionTransformer#,SaveModel
from tsai.all import get_tabular_dls


from fastai.data.transforms import Categorize
from fastai.tabular.core import Categorify, FillMissing, Normalize
#from fastai.tabular.all import * #TabularPandas, RandomSplitter, CrossEntropyLossFlat, tabular_learner

from astra.utils import cfg, logger, get_base_df ,align_dataframes, find_columns_with_word, clear_mem
from astra.data.datasets import TSDS
from astra.models.callbacks import CSaveModel
from astra.models.hybrid.mlm import *
from astra.visualize.evaluation import plot_evaluation, plot_box_kde

def pretrain(cfg):
    # Load data
    base = get_base_df()
    tsds = TSDS(cfg, base)
    
    concepts= ['VitaleVaerdier', 'Labsvar', 'Medicin','ITAOversigtsrapport']#, 'Procedurer']
    #concepts= ['VitaleVaerdier']
    holdout = TSDS(cfg, base[base.ServiceDate >='06-01-2023'])
    trainval = TSDS(cfg, base[base.ServiceDate <='06-01-2023'])

    trainval.complete = pd.concat(trainval.concepts).fillna(0.0)
    holdout.complete = pd.concat(holdout.concepts).fillna(0.0)
    trainval.complete, holdout.complete = align_dataframes(trainval.complete, holdout.complete)

    # Setup
    exclude = ['deceased_30d']


    cat_cols = cfg["dataset"]["cat_cols"]
    num_cols = cfg["dataset"]["num_cols"]
    
    logger.info(f'Categoricals: {cat_cols}\nNumericals: {num_cols}')

    # HOLDOUT 

    # Note, no batch shuffle on dataloder
    logger.info('Preparing holdout dataloader')
    tfms = [None, [Categorize()]]
    batch_tfms = TSStandardize(by_var=True)
    procs = [Categorify, FillMissing, Normalize]
    # Holdout set
    tX, ty = df2xy(holdout.complete, 
                   sample_col='PID', 
                   feat_col='FEATURE', 
                   data_cols=holdout.complete.columns[3:], 
                   target_col=holdout.target)
    ty = ty[:, 0].flatten()
    ty = list(ty)
    logger.info(f'Holdout X shape: {tX.shape}')
    test_ts_dls = get_ts_dls(tX, ty, splits=None, tfms=tfms, batch_tfms=batch_tfms, bs=cfg["training"]["bs"], drop_last=False,  shuffle=False)

    test_tab_dls = get_tabular_dls(holdout.tab_df, procs=procs, 
                                   cat_names=cat_cols.copy(), 
                                   cont_names=num_cols.copy(), 
                                   y_names=cfg["target"], 
                                   splits=None, 
                                   drop_last=False,
                                  shuffle=False)


    holdout_mixed_dls = get_mixed_dls(test_ts_dls, test_tab_dls, bs=cfg["training"]["bs"], shuffle_valid=False)

    # FOR CLASSES!
    complete_tab_dls = get_tabular_dls(pd.concat([trainval.tab_df, holdout.tab_df]), procs=procs, 
                               cat_names=cat_cols.copy(), 
                               cont_names=num_cols.copy(), 
                               y_names=cfg["target"], 
                               splits=None, 
                               drop_last=False,
                              shuffle=False)
    classes = complete_tab_dls.classes


    # Training and validation dataset
    logger.info("Setting up X,y for training and validation")
    X, y = df2xy(trainval.complete, 
             sample_col='PID', 
             feat_col='FEATURE', 
             data_cols=trainval.complete.columns[3:], 
             target_col=cfg["target"])
    y = y[:, 0].flatten()
    y = list(y)
    logger.info(f'Train/val X shape: {X.shape}')

    # TRAINING
    logger.info('Preparing training cycle')
    ts_dls = get_ts_dls(X, y, splits=None, tfms=tfms, batch_tfms=batch_tfms, bs=cfg["training"]["bs"], drop_last=False)

    tab_dls = get_tabular_dls(trainval.tab_df, procs=procs, 
                              cat_names=cat_cols.copy(), 
                              cont_names=num_cols.copy(), 
                              y_names=cfg["target"], 
                              splits=None, bs=cfg["training"]["bs"],
                              drop_last=False)

    mixed_dls = get_mixed_dls(ts_dls, tab_dls, 
                              bs=cfg["training"]["bs"])

    ####

    # 1. Create configuration
    config = MLMConfig(
        mask_prob_ts=0.10,
        mask_prob_cat=0.15,
        mask_prob_cont=0.15,
        epochs=30,
        lr=1e-5,
        warmup_epochs=3,
        ts_loss_weight=1.0,
        cat_loss_weight=1.0,
        cont_loss_weight=1.0,
        contrastive_weight=1.0,  # Enable contrastive learning
        patience=3,
        
        save_best=True,
        checkpoint_dir=f'./pretrain_checkpoints/{cfg["model_name"]}'
    )

    # Save config
    config.to_yaml('mlm_config.yaml')

    # Or load from file
    # config = MLMConfig.from_yaml('mlm_config.yaml')

    # 2. Pre-train
    backbone = TSTabFusionTransformer(c_in = mixed_dls.vars, c_out = 2,
                                          seq_len =  mixed_dls.len, classes=classes, cont_names= num_cols, 
                                           n_layers = 8,
                                           n_heads = 8,
                                           d_model=64,
                                           fc_dropout=0.75, 
                                           res_dropout=0.22,
                                          fc_mults = (0.3, 0.1),
                                  )
    mlm_model = TSTabFusionMLM(backbone, config)

    ts_splits = get_splits(y, 
                            valid_size=0.2, 
                          #  test_size=0.1,
                            stratify=True, 
                            random_state=42, 
                            shuffle=True, check_splits= True, show_plot=True)

    valid_idxs = list(ts_splits[1])
    n_val_idxs = len(valid_idxs)
    #logger.info(f"{n_val_idxs} in valid dataset with {self.y[valid_idxs].sum()} positive outcomes")

    # define new splits 
    splits = ((ts_splits[0],)+ (ts_splits[1],))


    ts_dls_ul = get_ts_dls(X, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=cfg["training"]["bs"], drop_last=False)

    tab_dls_ul = get_tabular_dls(trainval.tab_df, procs=procs, 
                              cat_names=cat_cols.copy(), 
                              cont_names=num_cols.copy(), 
                           #   y_names=cfg["target"], 
                              splits=splits, bs=cfg["training"]["bs"],
                              drop_last=False)

    mixed_dls_ul = get_mixed_dls(ts_dls_ul, tab_dls_ul, 
                              bs=cfg["training"]["bs"])

    history = pretrain_mlm_enhanced(
        mlm_model, 
        train_loader=mixed_dls_ul.train,
        val_loader=mixed_dls_ul.valid,  # Optional validation loader
        config=config,
        device='cuda'
    )
    logger.info('pretrained model saved')

def load_pretrained_backbone(path=f'pretrain_checkpoints/{cfg["model_name"]}/best_model.pt'):
    checkpoint = torch.load(path)
    return mlm_model.load_state_dict(checkpoint['model_state_dict'])
 
if __name__ == '__main__':
    pretrain(cfg)
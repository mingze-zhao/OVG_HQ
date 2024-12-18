"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

Moment-DETR (https://github.com/jayleicn/moment_detr)
Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
os.environ["WANDB_MODE"]="offline"
import sys
import time
import json
import pprint
import random
import argparse
import copy
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,pythonpath)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from easydict import EasyDict

from training.config import BaseOptions
from training.dataset_online import StartEndDataset, start_end_collate, prepare_batch_inputs, get_relevant_video_feat
# from training.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from training.cg_detr_dataset import CGDETR_StartEndDataset, cg_detr_start_end_collate, cg_detr_prepare_batch_inputs
from training.evaluate import eval_epoch, start_inference, setup_model

from lighthouse.common.utils.basic_utils import AverageMeter, dict_to_markdown, write_log, save_checkpoint, rename_latest_to_best, save_sh_n_codes, metricstocsv
from lighthouse.common.utils.model_utils import count_parameters, ModelEMA

from lighthouse.common.loss_func import VTCLoss
from lighthouse.common.loss_func import CTC_Loss
import wandb

import logging
import sys
import shutil
from pyinstrument import Profiler
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def additional_trdetr_losses(model_inputs, outputs, targets, opt):
    # TR-DETR only loss
    src_txt_mask,   src_vid_mask = model_inputs['src_txt_mask'], model_inputs['src_vid_mask']
    pos_mask =  targets['src_pos_mask'] 

    src_txt_ed, src_vid_ed =  outputs['src_txt_ed'], outputs['src_vid_ed']
    loss_align = CTC_Loss()
    loss_vid_txt_align = loss_align(src_vid_ed, src_txt_ed, pos_mask, src_vid_mask, src_txt_mask)

    src_vid_cls_ed = outputs['src_vid_cls_ed']
    src_txt_cls_ed = outputs['src_txt_cls_ed']
    loss_align_VTC = VTCLoss()
    loss_vid_txt_align_VTC = loss_align_VTC(src_txt_cls_ed, src_vid_cls_ed)

    loss = opt.VTC_loss_coef * loss_vid_txt_align_VTC + opt.CTC_loss_coef * loss_vid_txt_align
    return loss

def calculate_taskweave_losses(loss_dict, weight_dict, hd_log_var, mr_log_var):
    # TaskWeave only loss
    grouped_losses = {"loss_mr": [], "loss_hd": []}
    for k in loss_dict.keys():
        if k in weight_dict:
            if any(keyword in k for keyword in ["giou", "span", "label",'class_error']):
                grouped_losses["loss_mr"].append(loss_dict[k])
            elif "saliency" in k:
                grouped_losses["loss_hd"].append(loss_dict[k])
    loss_mr = sum(grouped_losses["loss_mr"])
    loss_hd = sum(grouped_losses["loss_hd"])
    # hd_log_var, mr_log_var = hd_log_var.to(loss_hd.device), mr_log_var.to(loss_mr.device)
    losses = 2 * loss_hd * torch.exp(-hd_log_var) + 1 * loss_mr * torch.exp(-mr_log_var) + hd_log_var + mr_log_var
    return losses

def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, teacher_model=None):
    batch_input_fn = cg_detr_prepare_batch_inputs  if opt.model_name == 'cg_detr' else prepare_batch_inputs
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    criterion.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    
    iter_modal_list = [None, "segment", "text"]
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        # continue
        break
        # if batch_idx == 100:
        #     break
        # profiler = Profiler()
        # profiler.start()
        model_inputs, targets = batch_input_fn(batch[1], opt.device, opt, training=True)

        if "video_feat" in opt.post_ttt_input_type:
            relevant_video_feat = get_relevant_video_feat(batch_meta=batch[0], video_feat_dict=train_loader.dataset.video_feat_dict)
        else:
            relevant_video_feat = None
        if opt.has_teacher:
            iter_modal_idx = batch_idx%len(iter_modal_list)
            use_modal = iter_modal_list[iter_modal_idx]
        else:
            use_modal=None
        if teacher_model is not None:
            teacher_outputs = teacher_model(**model_inputs, batch_meta = batch[0], relevant_video_feat = relevant_video_feat, training=True, use_modal="segment_teacher")
        outputs = model(**model_inputs, batch_meta = batch[0], relevant_video_feat = relevant_video_feat, training=True, use_modal = use_modal)
        if teacher_model is not None:
            outputs['teacher_pred_logits'] = teacher_outputs['pred_logits']
            outputs['teacher_pred_spans'] = teacher_outputs['pred_spans']
            outputs['teacher_saliency_scores'] = teacher_outputs['saliency_scores']
            outputs['teacher_query_feats'] = teacher_outputs['query_feats']
        
        loss_dict = criterion(outputs, targets, opt)        
        # remove reg_loss
        reg_loss = loss_dict['reg_loss'][1]
        # sal_loss = loss_dict['sal_loss'][1]
        loss_dict['reg_loss'] = loss_dict['reg_loss'][0]
        # loss_dict['sal_loss'] = loss_dict['sal_loss'][0]
        pos_cls_loss = loss_dict['cls_loss'][1]
        neg_cls_loss = loss_dict['cls_loss'][2]
        pos_cls_ratio = loss_dict['cls_loss'][3]
        neg_cls_ratio = loss_dict['cls_loss'][4]
        loss_dict['cls_loss'] = loss_dict['cls_loss'][0]

        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        loss_dict["loss_overall"] = float(losses)
        # profiler.stop()
        # profiler.print()

        loss_meters["loss_overall"].update(loss_dict["loss_overall"]) 
        loss_meters["cls_loss"].update(loss_dict['cls_loss'] * criterion.weight_dict['cls_loss'])
        if teacher_model is not None:
            loss_meters["dis_loss"].update(loss_dict['dis_loss'] * criterion.weight_dict['dis_loss'])
        loss_meters["cls_loss_pos"].update(pos_cls_loss, part=True) 
        loss_meters["cls_loss_neg"].update(neg_cls_loss, part=True) 
        loss_meters["pos_cls_ratio"].update(pos_cls_ratio) 
        loss_meters["neg_cls_ratio"].update(neg_cls_ratio) 
        loss_meters["reg_loss"].update(reg_loss* criterion.weight_dict['reg_loss'], part=True) 
        # loss_meters["sal_loss"].update(sal_loss* criterion.weight_dict['sal_loss'], part=True) 

        if batch_idx % opt.log_interval == 0:
            write_log(opt, epoch_i, loss_meters)
        
    # wandb.log({"train_loss_overall": loss_meters["loss_overall"].avg, 
    #             "train_loss_reg": loss_meters["reg_loss"].avg, 
    #             "train_loss_sal": loss_meters["sal_loss"].avg, 
    #             "train_loss_cls": loss_meters["cls_loss"].avg,
    #             "train_loss_cls_pos": loss_meters["cls_loss_pos"].avg,
    #             "train_loss_cls_neg": loss_meters["cls_loss_neg"].avg,
    #             "train_pos_cls_ratio": loss_meters["pos_cls_ratio"].avg, 
    #             "train_neg_cls_ratio": loss_meters["neg_cls_ratio"].avg,
    #             }, epoch_i)

def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt, teacher_model=None):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    collate_fn = cg_detr_start_end_collate if opt.model_name == 'cg_detr' else start_end_collate
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)

    # indices = np.random.choice(len(train_dataset), opt.bsz*4000, replace=False)
    # sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=opt.bsz,
        num_workers=0,
        shuffle=False,
        # pin_memory=True,
        # prefetch_factor=opt.bsz//4,
        # persistent_workers=True
        # sampler=sampler,
    )

    if opt.model_ema:
        logger.info("Using model EMA...")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = 0
    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, teacher_model = teacher_model)
        lr_scheduler.step()

        if opt.model_ema:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval == 0 and (epoch_i + 1) > opt.start_eval_epoch:
        # if True:
            if opt.has_teacher:
                # iter_modal_list = [None, "segment", "text"]
                iter_modal_list = ["text"]
            else:
                iter_modal_list = [None]
            for iter_modal in iter_modal_list:
                with torch.no_grad():
                    if opt.model_ema:
                        metrics, eval_loss_meters, latest_file_paths = \
                            eval_epoch(epoch_i, model_ema.module, val_dataset, opt, save_submission_filename, criterion,iter_modal=iter_modal)
                    else:
                        metrics, eval_loss_meters, latest_file_paths = \
                            eval_epoch(epoch_i, model, val_dataset, opt, save_submission_filename, criterion, iter_modal=iter_modal)

                write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode='val', iter_modal=iter_modal)     
                metricstocsv(epoch_i, metrics, opt.results_dir + f"metrics{iter_modal}.csv" if opt.results_dir[-1] == '/' else opt.results_dir +f"/metrics{iter_modal}.csv")       
                logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))
            
            # if opt.dset_name == 'tvsum' or opt.dset_name == 'youtube_highlight':
            #     stop_score = metrics["brief"]["mAP"]
            # else:
            #     stop_score = metrics["full"]["On3-R1@0.5-pos"]

        save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
            # if stop_score > prev_best_score:
            #     prev_best_score = stop_score
            #     best_epoch = epoch_i
            #     logger.info("The checkpoint file has been updated.")
            #     rename_latest_to_best(latest_file_paths)
            
            # if epoch_i == opt.n_epoch -1:
            #     best_file_path = os.path.join(opt.ckpt_filepath, f"epoch{best_epoch}.ckpt")
            #     new_file_path = os.path.join(opt.ckpt_filepath, "best.ckpt")
            #     os.rename(best_file_path, new_file_path)

def main(opt, yaml_path, pretrained_model_path, domain, debug):
    # dataset & data loader
    dataset_config = EasyDict(
        dset_name=opt.dset_name,
        domain=opt.domain,
        data_path=opt.train_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=opt.a_feat_dirs if "a_feat_dirs" in opt else [],
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=opt.a_feat_types if "a_feat_types" in opt else None,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
        segment_size=opt.segment_size,
        anchor_windows=opt.anchor_windows,
        use_online=opt.use_online,
        pos_threshold=opt.pos_threshold,
        label_file_path=opt.label_file_path,
        debug=debug,
        opt=opt,
        training=True,
    )

    train_dataset = CGDETR_StartEndDataset(**dataset_config) if opt.model_name == 'cg_detr' else StartEndDataset(**dataset_config)    
    copied_eval_config = copy.deepcopy(dataset_config)
    copied_eval_config.data_path = opt.eval_path
    copied_eval_config.q_feat_dir = opt.t_feat_dir_eval if "t_feat_dir_eval" in opt else opt.t_feat_dir
    copied_eval_config.training = False
    eval_dataset = CGDETR_StartEndDataset(**copied_eval_config) if opt.model_name == 'cg_detr' else StartEndDataset(**copied_eval_config)
    
    # prepare model
    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")

    # prepare teacher model
    if opt.has_teacher:
        teacher_model, criterion, _, _ = setup_model(opt)
        checkpoint = torch.load(opt.teacher_model_path)
        teacher_model.load_state_dict(checkpoint["model"], strict=False)
    # load checkpoint
    if pretrained_model_path is not None:
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint["model"])
        logger.info("Model checkpoint: {}".format(pretrained_model_path))
    count_parameters(model)
    logger.info("Start Training...")
    
    # start training
    if opt.has_teacher:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt, teacher_model = teacher_model)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='yaml config path for training. e.g., configs/qd_detr_qvhighlight.yml')
    parser.add_argument('--pretrained_model_path', type=str, help='saved model path', default=None)
    parser.add_argument('--domain', type=str, help='training domain for TVSum and YouTube Highlights . e.g., BK and dog. Note that they are not necessary for other datasets')
    parser.add_argument('--debug', type=bool, help='debug', default=False)
    parser.add_argument('--savecode', action='store_true', help='debug')
    args = parser.parse_args()
    yaml_path = args.config
    pretrained_model_path = args.pretrained_model_path
    domain = args.domain
    debug = args.debug

    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse(yaml_path, domain)
    opt["debug"] = debug
    if debug:
        opt.results_dir = "/mnt/data/jiaqi/online-vg/exp/debug"
        opt.wandb_dir = "/mnt/data/jiaqi/online-vg/wandb/debug"
        # opt.eval_path = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/highlight_val_release_debug.jsonl"
        opt.ckpt_filepath = "/mnt/data/jiaqi/online-vg/exp/debug/checkpoint"
        opt.start_eval_epoch = 0
        opt.start_eval_epoch = 0
    if args.savecode:
        save_sh_n_codes(opt.results_dir)
        if not os.path.exists(os.path.join(opt.results_dir, args.config.split('/')[-1])):
            shutil.copy(args.config, opt.results_dir)
    set_seed(opt.seed)
    if not os.path.exists(opt.wandb_dir):
        os.makedirs(opt.wandb_dir)
    # wandb.init(dir=opt.wandb_dir, project="online-vg", config=opt, name=opt.results_dir.replace("/mnt/data/jiaqi/online-vg/exp/", ""), entity="maojiaqics")
    main(opt, yaml_path, pretrained_model_path, domain, debug)

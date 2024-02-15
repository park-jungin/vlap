import argparse
import math
import os
import sys
import shutil
import time
import pickle
import json
import h5py
from tqdm import tqdm
from logging import getLogger
from typing import Tuple, Optional, Union

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from apex.parallel.LARC import LARC

from enum import Enum
from transformers import GPT2Tokenizer, GPTJModel, GPT2Model, GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM
from transformers import OPTModel, OPTForCausalLM
import transformers
from transformers import AutoTokenizer, AutoProcessor, CLIPVisionModel, CLIPTextModel
from transformers import T5EncoderModel, T5Tokenizer, T5Model, T5ForConditionalGeneration
from transformers import AutoImageProcessor, BeitModel, BeitFeatureExtractor
from transformers import AdamW, get_linear_schedule_with_warmup
import ftfy
import regex as re
import random
from PIL import Image

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from evalfunc.tokenizer.ptbtokenizer import PTBTokenizer
from evalfunc.bleu.bleu import Bleu
from evalfunc.rouge.rouge import Rouge
from evalfunc.cider.cider import Cider
from evalfunc.meteor.meteor import Meteor
from evalfunc.spice.spice import Spice

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class CaptionDataset(Dataset):

    def __init__(self, args, data_path: str, split: str, normalize_prefix=False):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.h = h5py.File(os.path.join(data_path, self.split + '_IMAGES_' + args.dataset + '_5_cap_per_img_5_min_word_freq' + '.hdf5'), 'r')
        print("====== Data is loaded ======")
        self.imgs = self.h['images']
        self.cpi = self.h.attrs['captions_per_image']

        with open(os.path.join(data_path, self.split + '_SENTENCES_' + args.dataset + '_5_cap_per_img_5_min_word_freq' + '.json'), 'r') as j:
            self.sentences = json.load(j)
        with open(os.path.join(data_path, self.split + '_CAPTIONS_' + args.dataset + '_5_cap_per_img_5_min_word_freq' + '.json'), 'r') as j:
            self.captions = json.load(j)
        with open(os.path.join(data_path, self.split + '_CAPLENS_' + args.dataset + '_5_cap_per_img_5_min_word_freq' + '.json'), 'r') as j:
            self.caplens = json.load(j)

        self.prefix_length = args.prefix_length
        self.max_seq_len = 80
        
        if args.image_encoder == 'clip':
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", device_map="auto")
        elif args.image_encoder == 'beit':
            self.processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k", device_map="auto")
        else:
            raise ValueError(f"Image model {args.image_encoder} not recognized")


        if args.text_encoder == 't5':
            self.caption_tokenizer = AutoTokenizer.from_pretrained("t5-base", 
                                                                    truncation=True, 
                                                                    padding="max_length", 
                                                                    max_length=self.max_seq_len, 
                                                                    device_map="auto")
        elif args.text_encoder == 'opt':
            self.caption_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", 
                                                                    truncation=True, 
                                                                    padding="max_length", 
                                                                    max_length=self.max_seq_len, 
                                                                    device_map="auto")
        else:
            raise ValueError(f"Text model {args.text_encoder} not recognized")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        img = self.imgs[item // self.cpi]
        img = np.array(np.transpose(np.uint8(img), (1, 2, 0)))
        img = Image.fromarray(img).convert('RGB')
        img = self.processor(images=img, return_tensors="pt")
        caption = ''
        for word in range(len(self.sentences[item])):
            if word == 0:
                caption = str(self.sentences[item][word])
            else:
                caption = caption + ' ' + str(self.sentences[item][word])
        caption_token = self.caption_tokenizer(caption, return_tensors="pt")
        input_ids = self.caption_tokenizer("A picture of ", 
                                            return_tensors="pt", 
                                            max_length=self.max_seq_len,
                                            truncation=True)
        label_caption = caption
        labels = self.caption_tokenizer(label_caption, 
                                            return_tensors="pt", 
                                            max_length=self.max_seq_len,
                                            truncation=True).input_ids
        token_no_pad = caption_token
        padding = self.max_seq_len - caption_token['input_ids'].shape[1]
        padding_label = self.max_seq_len - labels.shape[1]
        if padding > 0:
            caption_token['input_ids'] = torch.cat((caption_token['input_ids'], torch.zeros((1, padding), dtype=torch.int64)), 1)
            caption_token['attention_mask'] = torch.cat((caption_token['attention_mask'], torch.zeros((1, padding), dtype=torch.int64)), 1)
        else:
            caption_token['input_ids'] = caption_token['input_ids'][:, :self.max_seq_len]
            caption_token['attention_mask'] = caption_token['attention_mask'][:, :self.max_seq_len]
        if padding_label > 0:
            labels = torch.cat((labels, torch.zeros((1, padding_label), dtype=torch.int64)), 1)
        else:
            labels = labels[:, :self.max_seq_len]

        labels[labels == 0] = -100

        if self.split == 'VAL':
            return img, caption, input_ids
        else:
            return img, caption_token, token_no_pad, caption, input_ids, labels#, txt_feat


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class VLAP(nn.Module):

    def __init__(self, args, clip_length: Optional[int] = None, prefix_size: int = 768):
        super(VLAP, self).__init__()

        if args.text_encoder == 'gpt2':
            prefix_size = 768
        elif args.text_encoder == 't5':
            prefix_size = 512
        elif args.text_encoder == 'opt':
            prefix_size = 512

        self.prefix_size = prefix_size
        if args.image_encoder == 'clip':
            self.img_dim = 768
            self.img_projection = nn.Linear(self.img_dim, prefix_size, bias=True)
        elif args.image_encoder == 'beit':
            self.img_dim = 768
            self.img_projection = nn.Linear(self.img_dim, prefix_size, bias=True)
        
    def forward(self, emb_v, emb_t, tokens, vocab_emb, vocab_dist):
        bs = emb_v.size(0)
        if len(emb_v.shape) == 3:
            Ns = emb_v.size(1)
        else:
            Ns = 1
        Ns = 1
        emb_v = self.img_projection(emb_v)
        
        vocab_emb = nn.functional.normalize(vocab_emb.weight, dim=-1, p=2) # K x D
        vocab_size = vocab_emb.size(0)
        code_v = nn.functional.normalize(emb_v.mean(1), dim=-1, p=2)
        
        if Ns != 1:
            code_v = torch.mm(emb_v.reshape(-1, self.prefix_size), vocab_emb.t()) # B*N_s x K
            code_v = code_v.reshape(bs, Ns, -1)
        else:
            code_v = torch.mm(code_v, vocab_emb.t()) # B x K
        code_t = torch.mm(nn.functional.normalize(emb_t, dim=-1, p=2), vocab_emb.t()) # B x K

        loss = 0.

        with torch.no_grad():
            q_v = distributed_sinkhorn(code_v.detach(), vocab_dist)#[-bs:]   # B x K
            q_t = distributed_sinkhorn(code_t.detach(), vocab_dist)#[-bs:]   # B x K
            
        
        subloss = 0.
        if Ns != 1:
            x_v = code_v.mean(1) / args.temperature
        else:
            x_v = code_v / args.temperature
        x_t = code_t / args.temperature
        subloss -= torch.mean(torch.sum(q_t * F.log_softmax(x_v, dim=1), dim=1))
        subloss -= torch.mean(torch.sum(q_v * F.log_softmax(x_t, dim=1), dim=1))
        loss = subloss / 2

        return loss, emb_v, emb_t


logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of MVLM")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="./data/cc3m/train_val_test",
                    help="path to dataset repository")
parser.add_argument("--test_data_path", type=str, default="./data/coco/train_val_test",
                    help="path to dataset repository")
parser.add_argument("--dataset", type=str, default="coco",
                    help="dataset name choice: ['coco', 'nocap']")
parser.add_argument("--num_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6]")

##############################
#### MVLM specific params ####
##############################
parser.add_argument("--assign", type=int, nargs="+", default=[0, 1],
                    help="list of modality id used for computing assignments")
parser.add_argument("--hidden_dim", type=int, default=512,
                    help="hidden layer dimension from pretrained model, CLIP: 512, GPT: 768")
parser.add_argument("--temperature", type=float, default=0.01,
                    help="temporature parameter in training loss")
parser.add_argument("--epsilon", type=float, default=0.01,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")

                    
parser.add_argument("--sinkhorn_iterations", type=int, default=3,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--vocab_size", type=int, default=32128,
                    help="number of words in vocabulary: [50257 for gpt2, 32128 for t5, 50272 for opt]")
parser.add_argument("--feat_dim", default=512, type=int,
                    help="feature dimension (should be matched on both modalities)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from thie epoch, we start using a queue")
parser.add_argument("--image_encoder", type=str, default='beit',
                    help="pretrained encoder for visual representation choice: ['clip', 'beit']")
parser.add_argument("--text_encoder", type=str, default='t5',
                    help="pretrained encoder for text representation choice: ['t5', 'opt']")
parser.add_argument("--prefix_length", type=int, default=10,
                    help="prefix length for a caption model")
parser.add_argument("--mapping_type", type=str, default='mlp')

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", type=int, default=32,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", type=int, default=256,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", type=float, default=1e-2,  # 1e-2 for T5 1e-4 for opt
                    help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0,
                    help="final learning rate")
parser.add_argument("--warmup_epochs", default=3, type=int,
                    help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')

#########################
#### other parameters ###
#########################
parser.add_argument("--workers", default=16, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=1,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=False,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default="./checkpoint",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    device = torch.device(args.device)

    # build data
    train_dataset = CaptionDataset(args, data_path=args.data_path, split='TRAIN')
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, sampler=sampler, pin_memory=True, drop_last=True)
    val_dataset = CaptionDataset(args, data_path=args.test_data_path, split='VAL')
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, sampler=val_sampler, pin_memory=True)
    logger.info("Building data done.")

    # define model

    if args.text_encoder == 't5':
        TextModel = T5EncoderModel.from_pretrained("t5-base")
        CaptionModel = T5ForConditionalGeneration.from_pretrained('t5-base')
    elif args.text_encoder == 'opt':
        TextModel = OPTModel.from_pretrained("facebook/opt-1.3b")
        CaptionModel = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")

    if args.image_encoder == 'clip':
        VisionModel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    elif args.image_encoder == 'beit':
        VisionModel = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
    
    VisionModel, TextModel, CaptionModel = VisionModel.to(device), TextModel.to(device), CaptionModel.to(device)
    print("Encoders loaded.")

    if args.text_encoder == 't5':
        vocab_embedding = TextModel.shared.to(device)
    elif args.text_encoder == 'opt':
        vocab_embedding = TextModel.decoder.embed_tokens.to(device)
    # print(vocab_embedding.weight.shape)
    model = VLAP(args=args)
    print("Model loaded.")

    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        VisionModel = nn.SyncBatchNorm.convert_sync_batchnorm(VisionModel)
        TextModel = nn.SyncBatchNorm.convert_sync_batchnorm(TextModel)
        CaptionModel = nn.SyncBatchNorm.convert_sync_batchnorm(CaptionModel)

    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU

    model = model.to(device)
    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    cudnn.benchmark = True

    print("Load vocab distribution")

    if args.text_encoder == 't5':
        with open(os.path.join('./data/cc3m/train_val_test', 'VOCAB_DIST_T5.json'), 'rb') as j:
            vocab_dist = pickle.load(j)
    elif args.text_encoder == 'opt':
        with open(os.path.join('./data/cc3m/train_val_test', 'VOCAB_DIST_OPT.json'), 'rb') as j:
            vocab_dist = pickle.load(j)
    vocab_dist = torch.FloatTensor(vocab_dist).unsqueeze(1).to(device)
    print("Vocab distribution loaded")

    for epoch in range(start_epoch, args.epochs):
        # train the network for one epoch
        logger.info("================= Starting epoch %i ... =============" % epoch)

        # # set sampler
        # train_loader.sampler.set_epoch(epoch)
        
    
        scores = train(train_loader, VisionModel, TextModel,
                        model, CaptionModel, optimizer, epoch, lr_schedule, vocab_embedding, vocab_dist, device)
        training_stats.update(scores)

        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )


def train(train_loader, VisionModel, TextModel, model, CaptionModel, optimizer, epoch, lr_schedule, vocab_embedding, vocab_dist, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    caption_losses = AverageMeter()
    opt_losses = AverageMeter()

    VisionModel.eval()
    TextModel.eval()
    model.train()
    CaptionModel.eval()
    end = time.time()

    for param in CaptionModel.parameters():
        param.requires_grad = False

    dist.barrier()

    for it, (imgs, captions, token_no_pad, sentences, input_ids, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]
        model.zero_grad()
        imgs['pixel_values'] = imgs['pixel_values'].squeeze().to(device, non_blocking=True)

        with torch.no_grad():
            vis_emb = VisionModel(**imgs).last_hidden_state
            bs = vis_emb.size(0)
            captions['input_ids'] = captions['input_ids'].squeeze().to(device, non_blocking=True)
            captions['attention_mask'] = captions['attention_mask'].squeeze().to(device, non_blocking=True)
            tokens = captions['input_ids'].detach()
            token_no_pad['input_ids'] = token_no_pad['input_ids'].squeeze().to(device, non_blocking=True)
            token_no_pad['attention_mask'] = token_no_pad['attention_mask'].squeeze().to(device, non_blocking=True)
            if args.text_encoder == 'clip':
                txt_emb = TextModel.get_text_features(**token_no_pad)
            else:
                txt_emb = TextModel(**token_no_pad).last_hidden_state#.mean(dim=1)
            txt_emb_ = txt_emb * token_no_pad['attention_mask'].unsqueeze(2)
            txt_emb_ = txt_emb_.mean(1).squeeze()

        opt_loss, emb_v, emb_t = model(vis_emb, txt_emb_, tokens, vocab_embedding, vocab_dist)

        input_ids['input_ids'] = input_ids['input_ids'].squeeze().to(device, non_blocking=True)
        input_ids['attention_mask'] = input_ids['attention_mask'].squeeze().to(device, non_blocking=True)
        encoder_inputs_embeds = TextModel(**input_ids).last_hidden_state
        
        labels = labels.squeeze().to(device, non_blocking=True)
        if len(emb_v.shape) != 3:
            emb_v = emb_v.unsqueeze(1)
        encoder_inputs_embeds = torch.cat([emb_v, encoder_inputs_embeds], 1)
        encoder_inputs_attention = torch.ones(encoder_inputs_embeds.size()[:-1], 
                                            dtype=torch.long, 
                                            device=encoder_inputs_embeds.device)
    
        if args.text_encoder == 't5':
            caption_loss = CaptionModel(inputs_embeds=encoder_inputs_embeds, 
                                        labels=labels, 
                                        ).loss
        elif args.text_encoder == 'opt':
            encoder_inputs_embeds = torch.cat([encoder_inputs_embeds, txt_emb], dim=1)
            encoder_inputs_attention = torch.cat([encoder_inputs_attention, token_no_pad['attention_mask']], dim=1)
            outputs = CaptionModel(inputs_embeds=encoder_inputs_embeds, 
                                        attention_mask=encoder_inputs_attention,
                                        )
            logits = outputs.logits
            logits = logits[:, -labels.size(1):, :]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            caption_loss = loss_fct(shift_logits.view(-1, args.vocab_size), shift_labels.view(-1))
        loss = 0.1 * opt_loss + caption_loss

        optimizer.zero_grad()

        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        losses.update(loss.item(), bs)        
        caption_losses.update(caption_loss.item(), bs)
        opt_losses.update(opt_loss.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % 250 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "OpT-Gap Loss {opt_loss.avg:.4f}\t"
                "Caption Loss {cap_loss.avg:.4f}\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    opt_loss=opt_losses,
                    cap_loss=caption_losses,
                    lr=optimizer.param_groups[0]['lr']
                )
            )
    return (epoch, losses.avg)
    

@torch.no_grad()
def distributed_sinkhorn(out, vocab_dist):
    if len(out.shape) == 3:
        B = out.shape[0] * args.world_size
        Ns = out.shape[1]
        K = out.shape[2]
        out = out.reshape(-1, K)
        Q = torch.exp(out / args.epsilon).t() 
        
        for it in range(args.sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True) / Ns
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q *= (vocab_dist / 2.5e-1).softmax(dim=0)
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        
    else:
        Q = torch.exp(out / args.epsilon).t()
        B = Q.shape[1] * args.world_size # number of samples to assign
        K = Q.shape[0] 

        for it in range(args.sinkhorn_iterations):
            Q *= (vocab_dist / 2.5e-1).softmax(dim=0)
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment Q: K x B
    return Q.t()    # B x K or B*Ns x K


if __name__ == "__main__":
    main()

"""
File: pretrain.py
-------------------
Pretrain the CodonTransformer model.

The dataset is a JSON file. You can use prepare_training_data from CodonData to
prepare the dataset. The repository README has a guide on how to prepare the
dataset and use this script.
"""

import argparse
import os
import json
from typing import Union, List, Any, Optional
import math
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, EPOCH_OUTPUT, STEP_OUTPUT
from torch.utils.data import DataLoader, random_split
from transformers import BigBirdConfig, BigBirdForMaskedLM, BertTokenizer
from transformers.models.big_bird.modeling_big_bird import BigBirdEmbeddings, BigBirdLMPredictionHead, \
    BigBirdOnlyMLMHead
from transformers.modeling_outputs import MaskedLMOutput
import torch.nn as nn

from CodonTransformer.CodonPrediction import predict_dna_sequence
from CodonTransformer.CodonJupyter import format_model_output
from CodonTransformer.CodonUtils import (
    MAX_LEN,
    NUM_ORGANISMS,
    TOKEN2MASK,
    TOKEN2INDEX,
    IterableJSONData,
    INDEX2IRANGE,
    start_codon_index,
    INDEX2TOKEN,
    STOP_SYMBOL
)
from CodonTransformer.CodonData import (
    ORGANISM2ID2
)

import re
import CAI
import matplotlib.pyplot as plt

MAX_LR=3e-5
WARM_UP=0.1

class MaskedTokenizerCollator:
    def __init__(self, tokenizer, prob=0.15):
        self.tokenizer = tokenizer
        self.prob = prob

    def __call__(self, examples):
        tokenized = self.tokenizer(
            [ex["codons"] for ex in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        batch_size = tokenized["input_ids"].shape[0]
        seq_len = tokenized["input_ids"].shape[1]
        species_index = torch.tensor([ex["organism"] for ex in examples])
        # repeat方法可能有错误
        # tokenized["token_type_ids"] = species_index.repeat(1, seq_len)
        tokenized["token_type_ids"] = species_index.unsqueeze(1).expand(batch_size, seq_len)
        # print(tokenized["token_type_ids"].shape)

        inputs = tokenized["input_ids"]
        targets = inputs.clone()
        tokenized["eval_labels"] = inputs.clone()

        prob_matrix = torch.full(inputs.shape, self.prob)
        prob_matrix[inputs < 5] = 0.0
        selected = torch.bernoulli(prob_matrix).bool()

        # 80% of the time, replace masked input tokens with respective mask tokens
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = torch.tensor(list(map(TOKEN2MASK.__getitem__, inputs[replaced].numpy())))

        # 10% of the time, we replace masked input tokens with random vector.
        randomized = (
                torch.bernoulli(torch.full(selected.shape, 0.5)).bool()
                & selected
                & ~replaced
        )
        random_idx = torch.randint(26, 90, inputs.shape, dtype=torch.long)
        inputs[randomized] = random_idx[randomized]

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)

        return tokenized


class plTrainHarness(pl.LightningModule):
    def __init__(self, model, tokenizer, learning_rate, warmup_fraction, config=None,
                 samples_count=None, batch_size=None, valid_interval=None, accumulation_steps=None, scaling=1):
        super().__init__()
        # 十分重要，确保hparams.yaml不为空
        # self.save_hyperparameters()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction
        self.samples_count = samples_count
        self.batch_size = batch_size

        # 绘图部分
        self.valid_interval = valid_interval
        self.accum = accumulation_steps * scaling
        self.corrs = [0 for i in range(NUM_ORGANISMS)]
        self.totas = [0 for i in range(NUM_ORGANISMS)]
        self.corrs_va = [0 for i in range(NUM_ORGANISMS)]
        self.totas_va = [0 for i in range(NUM_ORGANISMS)]

        self.fig, self.ax2 = plt.subplots(1, 2)
        self.fig.suptitle('Codon Accuracy and Loss (With Organism Categories)', fontsize=20, fontweight='bold')
        self.ax2[0].set_title('Total Accuracy and Loss')
        self.ax2[1].set_title('Accuracy By Organism (Accumulated Train, Epoch Valid)')
        plt.tight_layout()
        self.left_labels = ['train-loss', 'train-accuracy', 'valid-loss', 'valid-accuracy']
        self.right_labels = ['human(train)', 'mouse(train)', 'Ec(train)', 'Sac(train)', 'Pic(train)']
        self.right_labels_va = ['human(validation)', 'mouse(validation)', 'Ec(validation)', 'Sac(validation)',
                                'Pic(validation)']
        self.right_ref_labels = ['human(train&ref)', 'mouse(train&ref)', 'Ec(train&ref)', 'Sac(train&ref)',
                                 'Pic(train&ref)']
        self.right_ref_labels_va = ['human(validation&ref)', 'mouse(validation&ref)', 'Ec(validation&ref)',
                                    'Sac(validation&ref)', 'Pic(validation&ref)']
        self.stept, self.stepv = [], []
        self.tr_loss, self.va_loss = [], []
        self.tr_acc, self.va_acc = [], []
        self.trlline, = self.ax2[0].plot(self.stept, self.tr_loss, 'b-o', label=self.left_labels[0])
        self.traline, = self.ax2[0].plot(self.stept, self.tr_acc, 'c-s', label=self.left_labels[1])
        self.valline, = self.ax2[0].plot(self.stepv, self.va_loss, 'r-o', label=self.left_labels[2])
        self.vaaline, = self.ax2[0].plot(self.stepv, self.va_acc, 'm-s', label=self.left_labels[3])
        self.organism_traccs = [[] for i in range(NUM_ORGANISMS)]
        self.organism_trlines = [[] for i in range(NUM_ORGANISMS)]
        self.organism_vaaccs = [[] for i in range(NUM_ORGANISMS)]
        self.organism_valines = [[] for i in range(NUM_ORGANISMS)]
        self.organism_colors = ['k-*', 'r-*', 'g-*', 'b-*', 'y-*']
        self.organism_colors_va = ['k-o', 'r-o', 'g-o', 'b-o', 'y-o']
        self.organism_refcolors = ['k-^', 'r-^', 'g-^', 'b-^', 'y-^']
        self.organism_refcolors_va = ['k-s', 'r-s', 'g-s', 'b-s', 'y-s']

        assert (len(self.organism_colors) >= NUM_ORGANISMS)

        for i in range(NUM_ORGANISMS):
            self.organism_trlines[i], = self.ax2[1].plot(self.stept, self.organism_traccs[i], self.organism_colors[i])
            self.organism_valines[i], = self.ax2[1].plot(self.stepv, self.organism_vaaccs[i],
                                                         self.organism_colors_va[i])

        self.ax2[0].legend(loc='upper right')
        leg1 = self.ax2[1].legend(self.organism_trlines, self.right_labels, loc='upper right')
        self.ax2[1].legend(self.organism_valines, self.right_labels_va, loc='lower right')
        self.ax2[1].add_artist(leg1)

        # 物种分类（训练集和验证集）
        self.ocorrs = [0 for i in range(NUM_ORGANISMS)]
        self.ototas = [0 for i in range(NUM_ORGANISMS)]
        self.ocorrs_va = [0 for i in range(NUM_ORGANISMS)]
        self.ototas_va = [0 for i in range(NUM_ORGANISMS)]

        self.ofig, self.oax = plt.subplots(1, 1)
        self.ofig.suptitle('Organism Classification Accuracy', fontsize=20, fontweight='bold')
        plt.tight_layout()
        self.organism_trclacc, self.organism_vaclacc = [], []
        self.organism_trclline, self.organism_vaclline = [], []
        self.organism_trclaccs = [[] for i in range(NUM_ORGANISMS)]
        self.organism_trcllines = [[] for i in range(NUM_ORGANISMS)]
        self.organism_vaclaccs = [[] for i in range(NUM_ORGANISMS)]
        self.organism_vacllines = [[] for i in range(NUM_ORGANISMS)]
        self.organism_trclline, = self.oax.plot(self.stept, self.organism_trclacc, 'm-*')
        self.organism_vaclline, = self.oax.plot(self.stepv, self.organism_vaclacc, 'm-o')
        for i in range(NUM_ORGANISMS):
            self.organism_trcllines[i], = self.oax.plot(self.stept, self.organism_trclaccs[i], self.organism_colors[i])
            self.organism_vacllines[i], = self.oax.plot(self.stepv, self.organism_vaclaccs[i],
                                                        self.organism_colors_va[i])
        leg1 = self.oax.legend(self.organism_trcllines, self.right_labels, loc='upper right')
        leg2 = self.oax.legend(self.organism_vacllines, self.right_labels_va, loc='lower right')
        self.oax.legend([self.organism_trclline, self.organism_vaclline],
                        ['total organism classification accuracy(train)',
                         'total organism classification accuracy(validation)'],
                        loc='center right')
        self.oax.add_artist(leg1)
        self.oax.add_artist(leg2)

        # 预测序列的gc和cai（参考验证集和训练集）
        excel_data = pd.read_excel('Privileged_Codon_Frequency_Table.xlsx', index_col=0)
        # self.codon_table = {}
        self.codon_table={c:excel_data[c].to_dict() for c in excel_data.columns}
        self.id2organism = {v: k for k, v in ORGANISM2ID2.items()}
        # for k in excel_data.columns:
        #     self.codon_table[k] = dict(zip(excel_data.index.tolist(), excel_data[k].values.tolist()))
        # print(excel_data.index.tolist())
        assert ('GCC' in excel_data.index.tolist())
        self.evfig, self.evax2 = plt.subplots(1, 2)
        self.evfig.suptitle('CAI and GC Attributes of Organism', fontsize=20, fontweight='bold')
        self.evax2[0].set_title('CAI by Organism')
        self.evax2[1].set_title('GC by Organism')
        plt.tight_layout()
        self.trgc, self.vagc, self.trcai, self.vacai = [], [], [], []
        self.trgc_line, self.vagc_line, self.trcai_line, self.vacai_line = [], [], [], []
        self.ref_trgc, self.ref_trcai, self.ref_trgc_line, self.ref_trcai_line = [], [], [], []
        self.ref_vagc, self.ref_vacai, self.ref_vagc_line, self.ref_vacai_line = [], [], [], []

        # valiadation验证集的物种级别列表最后一项必须先是[0,0],绘图时转换成单元素浮点数
        self.trgcs = [[] for i in range(NUM_ORGANISMS)]
        self.vagcs = [[[0, 0]] for i in range(NUM_ORGANISMS)]
        self.trcais = [[] for i in range(NUM_ORGANISMS)]
        self.vacais = [[[0, 0]] for i in range(NUM_ORGANISMS)]
        self.ref_trgcs = [[] for i in range(NUM_ORGANISMS)]
        self.ref_vagcs = [[[0, 0]] for i in range(NUM_ORGANISMS)]
        self.ref_trcais = [[] for i in range(NUM_ORGANISMS)]
        self.ref_vacais = [[[0, 0]] for i in range(NUM_ORGANISMS)]
        self.trgc_lines = [[] for i in range(NUM_ORGANISMS)]
        self.vagc_lines = [[] for i in range(NUM_ORGANISMS)]
        self.trcai_lines = [[] for i in range(NUM_ORGANISMS)]
        self.vacai_lines = [[] for i in range(NUM_ORGANISMS)]
        self.ref_trgc_lines = [[] for i in range(NUM_ORGANISMS)]
        self.ref_vagc_lines = [[] for i in range(NUM_ORGANISMS)]
        self.ref_trcai_lines = [[] for i in range(NUM_ORGANISMS)]
        self.ref_vacai_lines = [[] for i in range(NUM_ORGANISMS)]

        self.trcai_line, = self.evax2[0].plot(self.stept, self.trcai, 'm-*')
        self.vacai_line, = self.evax2[0].plot(self.stepv, self.vacai, 'm-o')
        self.trgc_line, = self.evax2[1].plot(self.stept, self.trgc, 'm-*')
        self.vagc_line, = self.evax2[1].plot(self.stepv, self.vagc, 'm-o')
        self.ref_trcai_line, = self.evax2[0].plot(self.stept, self.ref_trcai, 'm-^')
        self.ref_vacai_line, = self.evax2[0].plot(self.stepv, self.ref_vacai, 'm-s')
        self.ref_trgc_line, = self.evax2[1].plot(self.stept, self.ref_trgc, 'm-^')
        self.ref_vagc_line, = self.evax2[1].plot(self.stepv, self.ref_vagc, 'm-s')

        for i in range(NUM_ORGANISMS):
            self.trcai_lines[i], = self.evax2[0].plot(self.stept, self.trcais[i], self.organism_colors[i])
            self.vacai_lines[i], = self.evax2[0].plot(self.stepv, [], self.organism_colors_va[i])
            self.trgc_lines[i], = self.evax2[1].plot(self.stept, self.trgcs[i], self.organism_colors[i])
            self.vagc_lines[i], = self.evax2[1].plot(self.stepv, [], self.organism_colors_va[i])
            self.ref_trcai_lines[i], = self.evax2[0].plot(self.stept, self.ref_trcais[i], self.organism_refcolors[i])
            self.ref_vacai_lines[i], = self.evax2[0].plot(self.stepv, [], self.organism_refcolors_va[i])
            self.ref_trgc_lines[i], = self.evax2[1].plot(self.stept, self.ref_trgcs[i], self.organism_refcolors[i])
            self.ref_vagc_lines[i], = self.evax2[1].plot(self.stepv, [], self.organism_refcolors_va[i])

        leg1 = self.evax2[0].legend(self.trcai_lines + self.ref_trcai_lines, self.right_labels + self.right_ref_labels,
                                    loc='upper right')
        leg2 = self.evax2[0].legend(self.vacai_lines + self.ref_vacai_lines,
                                    self.right_labels_va + self.right_ref_labels_va, loc='lower right')
        self.evax2[0].legend([self.trcai_line, self.vacai_line, self.ref_trcai_line, self.ref_vacai_line],
                             ['total accumulated CAI(train)', 'total CAI(validation)', 'CAI train-REF',
                              'CAI validation-REF'], loc='center right')
        self.evax2[0].add_artist(leg1)
        self.evax2[0].add_artist(leg2)

        leg1 = self.evax2[1].legend(self.trgc_lines + self.ref_trgc_lines, self.right_labels + self.right_ref_labels,
                                    loc='upper right')
        leg2 = self.evax2[1].legend(self.vagc_lines + self.ref_vagc_lines,
                                    self.right_labels_va + self.right_ref_labels_va, loc='lower right')
        self.evax2[1].legend([self.trgc_line, self.vagc_line, self.ref_trgc_line, self.ref_vagc_line],
                             ['total accumulated GC(train)', 'total GC(validation)', 'GC train-REF',
                              'GC validation-REF'], loc='center right')
        self.evax2[1].add_artist(leg1)
        self.evax2[1].add_artist(leg2)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        # 显式计算total_steps
        if self.samples_count and self.batch_size:
            steps_per_epoch = self.samples_count // self.batch_size
            print(
                f"手动计算steps_per_epoch: {steps_per_epoch} (样本总数: {self.samples_count}, 批次大小: {self.batch_size})")
            total_steps = steps_per_epoch * self.trainer.max_epochs

        elif self.train_dataloader and hasattr(self.train_dataloader, '__len__'):
            steps_per_epoch = len(self.train_dataloader)
            total_steps = steps_per_epoch * self.trainer.max_epochs
        else:
            total_steps = self.trainer.estimated_stepping_batches

        if total_steps <= 0:
            raise ValueError(f"Total steps must be positive, got {total_steps}")

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=self.warmup_fraction,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def cal_acc(self, preds, batch, corrs, totas, train=True):
        assert (len(corrs) == NUM_ORGANISMS)
        assert (len(totas) == NUM_ORGANISMS)
        preds = torch.argmax(preds, dim=2, keepdim=False)
        assert (preds.ndim == 2)

        no_prefix = batch["attention_mask"]
        # 计算总体和各物种预测结果（用于准确率）
        corr = (preds == batch["labels"]) & (no_prefix)
        tota = no_prefix & (batch["labels"] >= 0)

        if not train:
            for i in range(NUM_ORGANISMS):
                correct_i = ((batch["token_type_ids"] == i) & corr).float().sum()
                total_i = ((batch["token_type_ids"] == i) & tota).float().sum()
                corrs[i] += correct_i - 2 * (batch["token_type_ids"][:, 0] == i).float().sum().detach()
                totas[i] += total_i - 2 * (batch["token_type_ids"][:, 0] == i).float().sum().detach()
                correct = corr.float().sum() - 2 * len(batch["labels"])
                total = tota.float().sum() - 2 * len(batch["labels"])
                return correct, total
        else:
            for i in range(NUM_ORGANISMS):
                correct_i = ((batch["token_type_ids"] == i) & corr).float().sum()
                total_i = ((batch["token_type_ids"] == i) & tota).float().sum()
                corrs[i] += correct_i
                totas[i] += total_i

            correct = corr.float().sum()
            total = tota.float().sum()
            self.tr_acc.append(float(correct / total))
            for i in range(NUM_ORGANISMS):
                self.organism_traccs[i].append(float(self.corrs[i] / (self.totas[i] + 1e-3)))
                self.organism_trlines[i].set_data(self.stept, self.organism_traccs[i])

            self.trlline.set_data(self.stept, self.tr_loss)
            self.traline.set_data(self.stept, self.tr_acc)
            self.ax2[0].relim()
            self.ax2[0].autoscale_view()
            self.ax2[1].relim()
            self.ax2[1].autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def cal_oacc(self, opreds, answers, ocorrs, ototas, train=True):
        assert (opreds.ndim == 2 and answers.ndim == 1)
        assert (opreds.shape[1] == NUM_ORGANISMS)
        assert (len(ocorrs) == NUM_ORGANISMS and len(ototas) == NUM_ORGANISMS)
        opreds = torch.argmax(opreds, dim=1)
        correct = (opreds == answers).float().sum()
        total = len(answers)
        # 每次大验证前验证计数置0
        for i in range(NUM_ORGANISMS):
            ocorrs[i] += float(((opreds == answers) * (answers == i)).float().sum())
            ototas[i] += float((answers == i).float().sum())

        if train:
            for i in range(NUM_ORGANISMS):
                self.organism_trclaccs[i].append(self.ocorrs[i] / (self.ototas[i] + 1e-3))
                self.organism_trcllines[i].set_data(self.stept, self.organism_trclaccs[i])
            if self.organism_trclacc == []:
                self.organism_trclacc.append(float(correct / total))
            else:
                self.organism_trclacc.append(
                    (float(correct / total) + len(self.organism_trclacc) * self.organism_trclacc[-1]) / (
                                len(self.organism_trclacc) + 1))
            self.organism_trclline.set_data(self.stept, self.organism_trclacc)
            self.oax.relim()
            self.oax.autoscale_view()
            self.ofig.canvas.draw()
            self.ofig.canvas.flush_events()

        return correct, total

    def cal_gc_and_cai(self, preds2, types, train=True, focus=None):
        # assert (preds2.ndim == 2)
        preds2 = preds2.detach().cpu().numpy()
        pred_seqs = np.array(list(map(INDEX2TOKEN.__getitem__, preds2.flatten()))).reshape(preds2.shape)

        if train and focus is not None:
            focus = focus.detach().cpu().numpy()
            pred_seqs = np.where(focus >= 0, pred_seqs, '')
            temp_seqs = []
            for i in range(len(pred_seqs)):
                temp_seqs.append(''.join([item.split(STOP_SYMBOL)[-1] for item in pred_seqs[i]]))
            pred_seqs = temp_seqs

            marks = []
            ref_cais, ref_gcs = [], []
            for i in range(len(pred_seqs)):
                seq = np.array([INDEX2TOKEN[li].split(STOP_SYMBOL)[-1] if li >= 0 else '' for li in focus[i]])
                seq = ''.join(seq)
                if seq == '' or 'unk' in seq:
                    continue
                marks.append(i)
                ref_cais.append(
                    CAI.CAI(seq, weights=self.codon_table[self.id2organism[int(types[i])]], genetic_code=11))
                ref_gcs.append(len(re.sub(r'[^gc]+', '', seq)) / len(seq))

            cai_record = [0 for i in range(NUM_ORGANISMS)]
            gc_record = [0 for i in range(NUM_ORGANISMS)]
            for i in range(len(marks)):
                gc_record[types[marks[i]]] += ref_gcs[i]
                cai_record[types[marks[i]]] += ref_cais[i]
            for i in range(NUM_ORGANISMS):
                if len(self.ref_trgcs[i]) > 0:
                    self.ref_trgcs[i].append((self.ref_trgcs[i][-1] * len(self.ref_trgcs[i]) + gc_record[i] /
                                              ((types == i).float().sum().detach().cpu().numpy() + 1e-3)) / (
                                                     len(self.ref_trgcs[i]) + 1))
                    self.ref_trcais[i].append((self.ref_trcais[i][-1] * len(self.ref_trcais[i]) + cai_record[i] /
                                               ((types == i).float().sum().detach().cpu().numpy() + 1e-3)) / (
                                                      len(self.ref_trcais[i]) + 1))
                else:
                    self.ref_trgcs[i].append(gc_record[i] / ((types == i).float().sum().detach().cpu().numpy() + 1e-3))
                    self.ref_trcais[i].append(
                        cai_record[i] / ((types == i).float().sum().detach().cpu().numpy() + 1e-3))

            for i in range(NUM_ORGANISMS):
                self.ref_trgc_lines[i].set_data(self.stept, self.ref_trgcs[i])
                self.ref_trcai_lines[i].set_data(self.stept, self.ref_trcais[i])

            marks = []
            gcs, cais = [], []
            for i in range(len(pred_seqs)):
                if pred_seqs[i] == '':
                    continue
                marks.append(i)
                # print(f'seq:{pred_seqs[i]}')
                cais.append(
                    CAI.CAI(pred_seqs[i], weights=self.codon_table[self.id2organism[int(types[i])]], genetic_code=11))
                gcs.append(len(re.sub(r'[^gc]+', '', pred_seqs[i])) / len(pred_seqs[i]))

            cai_record = [0 for i in range(NUM_ORGANISMS)]
            gc_record = [0 for i in range(NUM_ORGANISMS)]
            for i in range(len(marks)):
                gc_record[types[marks[i]]] += gcs[i]
                cai_record[types[marks[i]]] += cais[i]
            for i in range(NUM_ORGANISMS):
                if len(self.trgcs[i]) > 0:
                    self.trgcs[i].append((self.trgcs[i][-1] * len(self.trgcs[i]) + gc_record[i] / (
                            (types == i).float().sum().detach().cpu().numpy() + 1e-3)) / (len(self.trgcs[i]) + 1))
                    self.trcais[i].append((self.trcais[i][-1] * len(self.trcais[i]) + cai_record[i] / (
                            (types == i).float().sum().detach().cpu().numpy() + 1e-3)) / (len(self.trcais[i]) + 1))
                else:
                    self.trgcs[i].append(gc_record[i] / ((types == i).float().sum().detach().cpu().numpy() + 1e-3))
                    self.trcais[i].append(cai_record[i] / ((types == i).float().sum().detach().cpu().numpy() + 1e-3))

            for i in range(NUM_ORGANISMS):
                self.trgc_lines[i].set_data(self.stept, self.trgcs[i])
                self.trcai_lines[i].set_data(self.stept, self.trcais[i])

            return np.mean(gcs), np.mean(cais), np.mean(ref_gcs), np.mean(ref_cais), pred_seqs

        elif focus is not None:
            temp_seqs = []
            for i in range(len(pred_seqs)):
                temp_seqs.append(
                    ''.join([li.split(STOP_SYMBOL)[-1] if f > 4 else '' for f, li in zip(focus[i], pred_seqs[i])]))
            pred_seqs = temp_seqs

            for i in range(len(pred_seqs)):
                seq = [INDEX2TOKEN[int(li)].split(STOP_SYMBOL)[-1] if li > 4 else '' for li in focus[i]]
                if 'unk' in seq:
                    continue
                seq = ''.join(seq)
                # print(f'seq: {seq}')
                self.ref_vacais[types[i]][-1][0] += 1
                self.ref_vacais[types[i]][-1][1] += (
                    CAI.CAI(seq, weights=self.codon_table[self.id2organism[int(types[i])]], genetic_code=11))
                self.ref_vagcs[types[i]][-1][0] += 1
                self.ref_vagcs[types[i]][-1][1] += (len(re.sub(r'[^gc]+', '', seq)) / len(seq))
            gc, cai = 0, 0
            for i in range(len(pred_seqs)):
                # print(f'seq:{pred_seqs[i]}')
                gc += len(re.sub(r'[^gc]+', '', pred_seqs[i])) / len(pred_seqs[i])
                cai += CAI.CAI(pred_seqs[i], weights=self.codon_table[self.id2organism[int(types[i])]], genetic_code=11)
                self.vacais[types[i]][-1][0] += 1
                self.vacais[types[i]][-1][1] += (
                    CAI.CAI(pred_seqs[i], weights=self.codon_table[self.id2organism[int(types[i])]], genetic_code=11))
                self.vagcs[types[i]][-1][0] += 1
                self.vagcs[types[i]][-1][1] += (len(re.sub(r'[^gc]+', '', pred_seqs[i])) / len(pred_seqs[i]))
            gc /= len(pred_seqs)
            cai /= len(pred_seqs)

            return cai, gc, pred_seqs

    def training_step(self, batch, batch_idx):
        self.model.bert.set_attention_type("block_sparse")
        batch = {k: v.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")) for k, v in
                 batch.items()}
        batch2 = {k: v.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")) for k, v in
                  batch.items() if k in self.model.forward.__code__.co_varnames}

        # print(batch.keys())
        outputs = self.model(**batch2)
        self.log_dict(
            dictionary={
                "loss": outputs.loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            prog_bar=True,
        )

        if batch_idx % self.accum == 0:
            t_len = len(self.stept)
            self.stept.append(t_len)
            assert isinstance(outputs, MaskedLMOutput), "outputs has invalid type"
            # self.tr_loss.append(outputs.loss.clone().detach().cpu().numpy())
            self.tr_loss.append(float(outputs.loss.detach()))
            preds = outputs.logits[0]
            self.cal_acc(preds, batch2, self.corrs, self.totas, True)

            # 物种分类准确率
            opreds = outputs.logits[1]
            ans = batch2["token_type_ids"][:, 0]
            self.cal_oacc(opreds, ans, self.ocorrs, self.ototas, True)

            # gc/cai指标
            gc, cai, ref_gc, ref_cai, pred_seqs = self.cal_gc_and_cai(torch.argmax(preds, dim=2),
                                                                      batch2.get("token_type_ids")[:, 0], True,
                                                                      batch2.get('labels'))
            self.trgc.append(gc if self.trgc == [] else
                             (len(self.trgc) * self.trgc[-1] + gc) / (len(self.trgc) + 1))
            self.trcai.append(cai if self.trcai == [] else
                              (len(self.trcai) * self.trcai[-1] + cai) / (len(self.trcai) + 1))
            self.ref_trgc.append(
                ref_gc if self.ref_trgc == [] else (len(self.ref_trgc) * self.ref_trgc[-1] + ref_gc) / (
                        len(self.ref_trgc) + 1))
            self.ref_trcai.append(ref_cai if self.ref_trcai == [] else
                                  (len(self.ref_trcai) * self.ref_trcai[-1] + ref_cai) / (len(self.ref_trcai) + 1))
            print(f'last-step ref_trcai: {ref_cai}')

            self.trgc_line.set_data(self.stept, self.trgc)
            self.trcai_line.set_data(self.stept, self.trcai)
            self.ref_trgc_line.set_data(self.stept, self.ref_trgc)
            self.ref_trcai_line.set_data(self.stept, self.ref_trcai)

            self.evax2[0].relim()
            self.evax2[0].autoscale_view()
            self.evax2[1].relim()
            self.evax2[1].autoscale_view()
            self.evfig.canvas.draw()
            self.evfig.canvas.flush_events()

        return outputs.loss

    def validation_step(self, batch, batch_idx):

        # batch={k:v.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")) for k,v in batch.items()}
        batch2 = {}

        for k in batch.keys():
            if k == 'input_ids':
                l = batch.get('eval_labels').cpu().numpy()
                batch2[k] = torch.tensor(list(map(TOKEN2MASK.__getitem__, l.flatten()))).reshape(l.shape).to(
                    torch.device("cuda:0"))
            elif k == 'labels':
                batch2[k] = batch['eval_labels'].to(torch.device('cuda:0'))
            elif k != 'eval_labels':
                batch2[k] = batch[k].to(torch.device('cuda:0'))

        with torch.no_grad():
            outputs = self.model(**batch2)
        loss = outputs.loss

        preds = outputs.logits[0]
        correct, total = self.cal_acc(preds, batch2, self.corrs_va, self.totas_va, False)
        opreds = outputs.logits[1]
        ocorrect, ototal = self.cal_oacc(opreds, batch2["token_type_ids"][:, 0], self.ocorrs_va, self.ototas_va, False)
        cai, gc, pred_seqs = self.cal_gc_and_cai(torch.argmax(preds, dim=2, keepdim=False),
                                                 batch2["token_type_ids"][:, 0], False, batch2.get('labels'))
        ref_cais, ref_gcs = [], []
        bp_acc, count = 0, 0
        for i in range(len(pred_seqs)):
            seq = ''.join(
                [INDEX2TOKEN[int(li)].split(STOP_SYMBOL)[-1] if li > 4 else '' for li in batch2.get('labels')[i]])
            if 'unk' in seq:
                continue
            ref_cais.append(
                CAI.CAI(seq, weights=self.codon_table[self.id2organism[int(batch2["token_type_ids"][i, 0])]],
                        genetic_code=11))
            ref_gcs.append(len(re.sub(r'[^gc]+', '', seq)) / len(seq))
            assert (len(pred_seqs[i]) == len(seq))
            for pi, si in zip(pred_seqs[i], seq):
                bp_acc += float(pi == si)
            count += len(seq)

        # 返回该批次的损失、正确预测数及样本数
        return {'val_loss': loss, 'correct': correct, 'total': total, 'ocorrect': ocorrect, 'ototal': ototal,
                'cai': cai, 'gc': gc, 'rcai': np.mean(ref_cais), 'rgc': np.mean(ref_gcs), 'bp_acc': bp_acc / count}

    def validation_epoch_end(self, validation_step_outputs):
        # 初始化累计变量
        total_loss = 0
        total_correct = 0
        total_codons = 0
        total_corro = 0
        total_organisms = 0
        total_gc, total_rgc = 0, 0
        total_cai, total_rcai = 0, 0
        total_bp_acc = 0

        # 聚合所有批次的输出
        for output in validation_step_outputs:
            total_loss += output['val_loss']
            total_correct += output['correct']
            total_codons += output['total']
            total_corro += output['ocorrect']
            total_organisms += output['ototal']
            total_gc += output['gc']
            total_cai += output['cai']
            total_rgc += output['rgc']
            total_rcai += output['rcai']
            total_bp_acc += output['bp_acc']

        # 计算整个验证集的平均损失和准确率
        avg_loss = total_loss / len(validation_step_outputs)
        accuracy = total_correct / total_codons
        oaccuracy = total_corro / total_organisms
        gc = total_gc / len(validation_step_outputs)
        cai = total_cai / len(validation_step_outputs)
        rgc = total_rgc / len(validation_step_outputs)
        rcai = total_rcai / len(validation_step_outputs)
        bp_acc = total_bp_acc / len(validation_step_outputs)

        # 使用 self.log 记录指标，on_epoch=True 表示记录整个epoch的平均值
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.log('val_codon_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.log('val_basepair_accuracy', bp_acc, on_epoch=True, prog_bar=True)
        self.log('\nInference GC', gc, on_epoch=True, prog_bar=True)
        self.log('Reference GC', rgc, on_epoch=True, prog_bar=True)
        self.log('Inference CAI', cai, on_epoch=True, prog_bar=True)
        self.log('Reference CAI', rcai, on_epoch=True, prog_bar=True)
        self.log('Organism Classification', oaccuracy, on_epoch=True, prog_bar=True)

        v_len = len(self.stepv)
        self.stepv.append(self.valid_interval // self.accum * (v_len + 1))
        self.va_loss.append(float(avg_loss))
        self.va_acc.append(float(accuracy))
        self.valline.set_data(self.stepv, self.va_loss)
        self.vaaline.set_data(self.stepv, self.va_acc)
        self.ax2[0].relim()
        self.ax2[0].autoscale_view()
        for i in range(NUM_ORGANISMS):
            self.organism_vaaccs[i].append(float(self.corrs_va[i] / (self.totas_va[i] + 1e-3)))
            self.organism_valines[i].set_data(self.stepv, self.organism_vaaccs[i])
        self.ax2[1].relim()
        self.ax2[1].autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # 物种分类
        for i in range(NUM_ORGANISMS):
            self.organism_vaclaccs[i].append(float(self.ocorrs_va[i] / (self.ototas_va[i] + 1e-3)))
            self.organism_vacllines[i].set_data(self.stepv, self.organism_vaclaccs[i])

        self.organism_vaclacc.append(float(oaccuracy))
        self.organism_vaclline.set_data(self.stepv, self.organism_vaclacc)
        self.oax.relim()
        self.oax.autoscale_view()
        self.ofig.canvas.draw()
        self.ofig.canvas.flush_events()

        # gc/cai
        self.vacai.append(float(cai))
        self.vacai_line.set_data(self.stepv, self.vacai)
        self.vagc.append(float(gc))
        self.vagc_line.set_data(self.stepv, self.vagc)
        self.ref_vacai.append(float(rcai))
        self.ref_vacai_line.set_data(self.stepv, self.ref_vacai)
        self.ref_vagc.append(float(rgc))
        self.ref_vagc_line.set_data(self.stepv, self.ref_vagc)

        for i in range(NUM_ORGANISMS):
            self.vagcs[i][-1] = self.vagcs[i][-1][1] / (self.vagcs[i][-1][0] + 1e-3)
            self.vacais[i][-1] = self.vacais[i][-1][1] / (self.vacais[i][-1][0] + 1e-3)
            self.ref_vagcs[i][-1] = self.ref_vagcs[i][-1][1] / (self.ref_vagcs[i][-1][0] + 1e-3)
            self.ref_vacais[i][-1] = self.ref_vacais[i][-1][1] / (self.ref_vacais[i][-1][0] + 1e-3)
            self.vacai_lines[i].set_data(self.stepv, self.vacais[i])
            self.vagc_lines[i].set_data(self.stepv, self.vagcs[i])
            self.ref_vacai_lines[i].set_data(self.stepv, self.ref_vacais[i])
            self.ref_vagc_lines[i].set_data(self.stepv, self.ref_vagcs[i])

        self.evax2[0].relim()
        self.evax2[0].autoscale_view()
        self.evax2[1].relim()
        self.evax2[1].autoscale_view()
        self.evfig.canvas.draw()
        self.evfig.canvas.flush_events()

        for i in range(NUM_ORGANISMS):
            self.corrs_va[i] = 0
            self.totas_va[i] = 0
            self.ocorrs_va[i] = 0
            self.ototas_va[i] = 0
            self.vagcs[i].append([0, 0])
            self.vacais[i].append([0, 0])
            self.ref_vagcs[i].append([0, 0])
            self.ref_vacais[i].append([0, 0])

        # 如果需要，也可以返回一个字典（可选，通常记录更重要）
        return {'val_loss': avg_loss, 'val_codon_accuracy': accuracy, 'val_bp_acc': bp_acc,
                'cai': cai, 'gc': gc, 'rcai': rcai, 'rgc': rgc, 'val_clacc': oaccuracy}



    # 取决于pl.LightningDataModule的数据迭代器
    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return {"protein": batch["protein"], "organism": batch["organism"]}

    def test_epoch_end(self, outputs) -> None:
        for step_output in outputs:
            output = predict_dna_sequence(
                protein=step_output["protein"],
                organism=step_output["organism"],
                device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
                tokenizer=self.tokenizer,
                model=self.model,
                attention_type="block_sparse",
                deterministic=True
            )
            print(format_model_output(output))


class EpochCheckpoint(pl.Callback):
    def __init__(self, checkpoint_dir, save_interval):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        os.makedirs(self.checkpoint_dir,exist_ok=True)
        if current_epoch % self.save_interval == 0 or current_epoch == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"epoch_{current_epoch}.ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"\nCheckpoint saved at {checkpoint_path}\n")


# 1. 创建自定义的Embedding类
class CustomBigBirdEmbeddings(BigBirdEmbeddings):
    def __init__(self, config, ko=1, kp=1, asym=False, step=1):
        super().__init__(config)
        self.asym = asym
        self.ko=ko
        self.kp = kp
        self.count = 0
        self.step = step

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                # 使用repeat实现，expand不好实现
                buffered_token_type_ids_expanded = buffered_token_type_ids.squeeze(0).repeat(input_shape[0], 1)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros((input_shape[0], input_shape[1]), dtype=torch.long,
                                             device=self.position_ids.device)

        assert (torch.max(token_type_ids) <= NUM_ORGANISMS - 1)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.asym:
            ko = self.ko + (1 - self.ko) * math.exp(-self.count)
            embeddings += ko * token_type_embeddings
        else:
            embeddings += self.ko * token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        if self.asym:
            kp = self.kp + (1 - self.kp) * math.exp(-self.count)
            embeddings += kp * position_embeddings
        else:
            embeddings += self.kp * position_embeddings

        self.count += self.step

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)

        return embeddings

    def get_token_type_embeddings(self):
        return self.token_type_embeddings

    def set_token_type_embeddings(self,new_embeddings):
        if isinstance(new_embeddings,nn.Module):
            self.token_type_embeddings=new_embeddings

class MyCodonDecoderHead(BigBirdLMPredictionHead):
    def __init__(self, config):
        super().__init__(config)
        self.decoder_organism_bias = nn.Parameter(torch.zeros(config.type_vocab_size,requires_grad=True))
        # waiting to be shared
        self.decoder_organism=nn.Linear(config.hidden_size,config.type_vocab_size,bias=False)
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        vocab = self.decoder(hidden_states)
        type_vocab = self.decoder_organism(hidden_states) + self.decoder_organism_bias
        # mean calculation on T-axis for BTV
        type_vocab = torch.mean(type_vocab, dim=1, keepdim=False)

        return vocab,type_vocab


    def get_decoder_organism(self):
        return self.decoder_organism

    def set_decoder_organism(self, new_embeddings):
        if isinstance(new_embeddings,nn.Module):
            print(self.decoder_organism.weight.shape)
            print(new_embeddings.weight.shape)
            assert(self.decoder_organism.weight.shape==new_embeddings.weight.shape)
            self.decoder_organism.weight = new_embeddings.weight




class MyCodonBBMLMHead(BigBirdOnlyMLMHead):
    def __init__(self, config):
        super().__init__(config)
        self.predictions = MyCodonDecoderHead(config)

    def forward(self, sequence_output):
        return self.predictions(sequence_output)


class MyDataLoader(pl.LightningDataModule):
    def __init__(self, train_dataset=None, valid_dataset=None, test_dataset=None, collate_fn=None, batch_size=6,
                 num_workers=0, persistent_workers=False, split_ratio=0.9):
        super().__init__()
        self.train_dataset = list(train_dataset)
        self.valid_dataset = list(valid_dataset)
        self.test_dataset = list(test_dataset)
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.split_ratio = 0.9

    def setup(self, stage=None):
        pass
        # 每轮随机划分数据集，随机划分，不考虑结构差异，可能训练-验证集数据泄露
        # if stage == 'fit' or stage is None:
        # size = len(list(self.tv_dataset))
        # self.train_dataset, self.val_dataset = random_split(list(self.tv_dataset), [int(self.split_ratio*size), size-int(self.split_ratio*size)])
        # self.train_dataset, self.val_dataset = list(self.tv_dataset)[:int(self.split_ratio * size)], list(
        #     self.tv_dataset)[int(self.split_ratio * size):]

    def train_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.collate_fn, num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)



class MyBigBirdModel(BigBirdForMaskedLM):
    # type_lf measures relative importance of organism information compared with AD-token information
    def __init__(self, config, ko=1, kp=1, scale=1, asymptotic=False, step=1, type_lf=1):
        super().__init__(config)
        self.asymptotic = asymptotic
        self.ko=ko
        self.kp = kp
        self.scalev = nn.Parameter(torch.tensor(scale,dtype=torch.bfloat16,requires_grad=True))
        self.scalet = nn.Parameter(torch.tensor(scale,dtype=torch.bfloat16,requires_grad=True))
        self.count = 0
        self.step = step
        self.type_lf=type_lf
        self.bert.embeddings=CustomBigBirdEmbeddings(config,self.ko,self.kp,self.asymptotic, self.step)
        self.cls = MyCodonBBMLMHead(config)
        self.cls.predictions.set_decoder_organism(self.bert.embeddings.get_token_type_embeddings())
        self.i2ir_pt = parallel_i2ir(INDEX2IRANGE)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores, prediction_organism_scores = self.cls(sequence_output)
        shape_store = prediction_scores.shape
        prediction_scores[:,0,start_codon_index]=10000
        prediction_scores = prediction_scores.reshape(-1, shape_store[-1])
        if labels is not None:
            labels = labels.reshape(-1)
            # for evaluation
            labels = torch.where(labels.view(-1) > 4, labels, -100)
            prediction_scores[self.i2ir_pt[torch.where(labels>=0,labels,0)]] = -10000
        # for inference
        else:
            prediction_scores[self.i2ir_pt[input_ids.reshape(-1)]] = -10000

        # scale放缩系数处理
        if self.asymptotic:
            scalev = self.scalev + (1 - self.scalev) * math.exp(-self.count)
            prediction_scores *= scalev
            scalet = self.scalet + (1 - self.scalet) * math.exp(-self.count)
            prediction_organism_scores *= scalet
        else:
            prediction_scores *= self.scalev
            prediction_organism_scores *= self.scalet

        self.count += self.step

        masked_lm_loss = None
        if labels is not None and token_type_ids is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            tlabels = token_type_ids[:, 0]
            masked_lm_loss = (loss_fct(prediction_scores, labels)
                              + self.type_lf * loss_fct(prediction_organism_scores.view(-1, self.config.type_vocab_size),
                                               tlabels.view(-1)))
            labels = labels.reshape(shape_store[:2])

        prediction_scores = prediction_scores.reshape(shape_store)

        if not return_dict:
            output = (prediction_scores, prediction_organism_scores,) + outputs[2:]
            # 返回非类对象形式的元组统计
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=(prediction_scores, prediction_organism_scores,),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main(args):
    """Pretrain the CodonTransformer model."""
    # pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    config = BigBirdConfig(
        vocab_size=len(tokenizer),
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=MAX_LEN,
        type_vocab_size=NUM_ORGANISMS,
        sep_token_id=2,
        block_size=32,
    )
    model = MyBigBirdModel(config=config, kp=1, scale=1, asymptotic=False, step=1, type_lf=0)

    harnessed_model = plTrainHarness(model, tokenizer, args.learning_rate, args.warmup_fraction, config,
                                     valid_interval=args.valid_check_interval,
                                     accumulation_steps=args.accumulate_grad_batches,scaling=15)
    if args.ckpt_path!="":
        if os.path.isfile(args.ckpt_path):
            harnessed_model=harnessed_model.load_from_checkpoint(args.ckpt_path)
        else:
            raise FileExistsError("Checkpoint file not existent.")

    # Load the training data
    train_data = IterableJSONData(args.train_data_path)
    test_data = IterableJSONData(args.test_data_path)
    print(len(list(test_data)))

    dataloader = MyDataLoader(
        train_dataset=train_data,
        valid_dataset=test_data,
        test_dataset=test_data,
        collate_fn=MaskedTokenizerCollator(tokenizer, prob=0.15),
        batch_size=args.batch_size,
        num_workers=0 if args.debug else args.num_workers,
        persistent_workers=False,
    )

    # Setup trainer and callbacks
    logger=CSVLogger(args.checkpoint_dir,name=None)
    save_checkpoint = EpochCheckpoint(args.checkpoint_dir, args.max_epochs-1)
    trainer = pl.Trainer(
        logger=logger,
        strategy="dp",
        accelerator="gpu",
        devices=1 if args.debug else args.num_gpus,
        precision="bf16",
        max_epochs=args.max_epochs,
        deterministic=False,
        enable_checkpointing=False,
        callbacks=[save_checkpoint],
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.valid_check_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=1.0,  # 只验证20%的数据，加快速度
        num_sanity_val_steps=0,  # 跳过初始验证
    )

    plt.ion()
    # Continue training the model(not equivalent to finetuning)
    # trainer.fit(model=harnessed_model, datamodule=dataloader, ckpt_path=args.ckpt_path if args.ckpt_path!="" else None)
    # Pretrain the model
    trainer.fit(model=harnessed_model, datamodule=dataloader)
    plt.ioff()
    # 维持曲线图
    plt.show()


def create_vocab_txt(vocab_dict, path=''):
    fpath = ""
    if path != '':
        os.makedirs(path, exist_ok=True)
        fpath = path + '/vocab.txt'
    else:
        fpath = "vocab.txt"

    with open(fpath, 'w', encoding="utf-8") as f:
        for k, v in sorted(vocab_dict.items(), key=lambda x: x[1]):
            f.write(f"{k}\n")

    return fpath


def create_vocab_json(vocab_dict, path=''):
    fpath = ""
    if path != '':
        os.makedirs(path, exist_ok=True)
        fpath = path + '/vocab.json'
    else:
        fpath = "vocab.json"

    with open(fpath, 'w', encoding="utf-8") as f:
        json.dump(vocab_dict, f, indent=2)

    return fpath


def fast_tokenizer_construct(vocab_dict, path=''):
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers import Tokenizer
    # 构建并从vocab.json加载词汇表
    fpath = create_vocab_txt(vocab_dict, path)

    # 从现有的vocab.txt文件创建

    tokenizer = BertTokenizer(fpath,
                              never_split=[k for k in vocab_dict.keys()],
                              unk_token="[UNK]",
                              sep_token="[SEP]",
                              pad_token="[PAD]",
                              cls_token="[CLS]",
                              mask_token="[MASK]")

    # tokenizer = BertTokenizer(fpath)
    tokenizer.pre_tokenizer = Whitespace()

    # tokenizer.model = WordLevel.from_file(fpath)

    # 设置后处理等（如果需要）
    """
    from tokenizers.processors import TemplateProcessing

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
            ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
        ],
    )"""

    dir = os.path.dirname(fpath)
    # 保存为tokenizer.json
    tokenizer.save_pretrained(dir)

    return dir


# parallel accelerated calculation
# directly padding -torch.inf!
def parallel_i2ir(i2ir_dict):
    I2IR_PT = torch.zeros(len(i2ir_dict), len(i2ir_dict))
    for i in range(5, len(i2ir_dict)):
        for j in range(len(i2ir_dict)):
            if j not in i2ir_dict[i]:
                I2IR_PT[i, j] = 1
    I2IR_PT = I2IR_PT.bool()
    return I2IR_PT.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


if __name__ == "__main__":
    path = fast_tokenizer_construct(TOKEN2INDEX, 'tokenizing')
    print(f'path:{path}')
    parser = argparse.ArgumentParser(description="Pretrain the revised BigBirdForMaskedLM model.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=False,
        default=path,
        help="Path to the tokenizer model file",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=False,
        default="dataset/training_data_0.9_0.3_0.8.jsonl",
        help="Path to the training data JSON file",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=False,
        default="dataset/valid_data_0.1_0.3_0.8.jsonl",
        help="Path to the testing data JSON file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        default="checkpoints",
        help="Directory where checkpoints will be saved",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--valid_check_interval", type=int, default=2000, help="validating period on batch iters"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=8,
        help="Number of batches to accumulate gradients",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=MAX_LR,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=WARM_UP,
        help="Fraction of total steps to use for warmup",
    )
    parser.add_argument(
        "--save_interval", type=int, default=2, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to a certain checkpoint file",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    # args.ckpt_path="checkpoints/lightning_logs/version_70/checkpoints/epoch=4-step=7003.ckpt"
    # print(INDEX2IRANGE.keys())
    # print(INDEX2IRANGE.items())
    main(args)

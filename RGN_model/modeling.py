# coding=utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from transformers import BertPreTrainedModel, RobertaModel, RobertaConfig

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
}

class RGN(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RGN, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        self.class_loss_fct = CrossEntropyLoss()
        self.consistency_loss_fct = L1Loss()

        ### gate function
        self.gate = nn.Linear(768, 1)
        self.rel_gate = nn.Linear(768, 1)
        # self.classifier = nn.Linear(768 * 10, 3)

    def forward(self, input_ids, attention_mask, token_type_ids,
                position_ids=None, head_mask=None, labels=None,
                labels_one_hot=None, aug_labels_one_hot=None, paired=False, triplet=False, top_k=10):

        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=None,
                               position_ids=position_ids,
                               head_mask=head_mask)

        # pred for original data as usual
        sequence_output = outputs[0]

        ### start gate function for q and doc
        # print(sequence_output.size())
        gate_format = sequence_output.view(-1, sequence_output.size()[2]) ## (bz * max_len, 768)
        gate_score = self.gate(gate_format) ## (bz * max_len, 1)
        sequence_output_gate = gate_score.view(sequence_output.size()[0], sequence_output.size()[1]) #### (bz, max_len)
        sequence_output_gate_softmax = F.softmax(sequence_output_gate, dim=1)
        ### choose top k
        top_k_score, topk_k_index = torch.topk(sequence_output_gate_softmax, top_k, sorted=False) ##(bz, topk)   
        ### retrieve the top k representations from roberta
        relevant_word_emb = []
        for i in range(0, sequence_output.size()[0]):
            tmp = torch.index_select(sequence_output[i], 0, topk_k_index[i]) ## 
            relevant_word_emb.append(tmp)
        lang_relat = torch.cat(relevant_word_emb, 0) ## [bz * topk, 768]
        lang_candidate_relations = lang_relat.view(sequence_output.size()[0], top_k, sequence_output.size()[2])
        lang_relat = lang_relat.view(sequence_output.size()[0], top_k * sequence_output.size()[2]) ## [bz, topk * 768] it should change in the later time.

        ## relation topk * (topk - 1) // 2
        lang_candidate_relations_1 = lang_relat.view(sequence_output.size()[0], top_k, 1,  sequence_output.size()[2])
        lang_candidate_relations_2 = lang_relat.view(sequence_output.size()[0], 1, top_k,  sequence_output.size()[2])
        lang_candidate_relations_repeat_1 = lang_candidate_relations_1.repeat(1, 1, top_k, 1) 
        lang_candidate_relations_repeat_2 = lang_candidate_relations_2.repeat(1, top_k, 1, 1) 
        lang_candidate_relations = lang_candidate_relations_repeat_1 + lang_candidate_relations_repeat_2
        lang_candidate_relations = lang_candidate_relations.view(sequence_output.size()[0], top_k * top_k, sequence_output.size()[2])

        # import pdb
        # pdb.set_trace()
        relate_ind = torch.tril_indices(top_k, top_k, -1).cuda()
        relate_ind[1] = relate_ind[1] * top_k
        relate_ind = relate_ind.sum(0)

        relate_stack = lang_candidate_relations.index_select(1, relate_ind)

        ## Old method to implement the relational gating. 
        # tmp_list = []
        # for i in range(lang_candidate_relations.size()[0]):
        #     tmp_tensor = torch.index_select(lang_candidate_relations[i], 0, relate_ind[i])
        #     tmp_list.append(tmp_tensor)
        # # relate_stack = lang_candidate_relations.index_select(1, relate_ind) ## [bz, topk * (topk - 1) // 2, 768]
        # relate_stack = torch.cat(tmp_list, 0) ## [bz * topk, 768]
        # relate_stack = relate_stack.view(sequence_output.size()[0], top_k * (top_k - 1) // 2, sequence_output.size()[2])

        ## New method to implement the relational gating.
        rel_gate_score = self.rel_gate(relate_stack) ## (bz * max_len, 1)
        relate_stack_rel_gate = rel_gate_score.view(relate_stack.size()[0], relate_stack.size()[1]) #### (bz, max_len)
        relate_stack_rel_gate_softmax = F.softmax(relate_stack_rel_gate, dim=1)

        ## choose top k
        top_k_rel_score, topk_k_rel_index = torch.topk(relate_stack_rel_gate_softmax, top_k, sorted=False) ##(bz, topk)   
        relate_emb = []
        for i in range(0, relate_stack.size()[0]):
            tmp = torch.index_select(relate_stack[i], 0, topk_k_rel_index[i]) 
            relate_emb.append(tmp)
        relation_rep = torch.cat(relate_emb, 0) ## [bz * topk, 768] 
        relation_rep = relation_rep.view(relate_stack.size()[0], top_k * relate_stack.size()[2]) ## [bz, topk * 768]

        lang_relat = torch.cat([lang_relat, relation_rep], 1) ## [bz, topk * 768 * 2]

        logits = self.classifier(lang_relat)  ### [bz, num_label]
        outputs =  F.softmax(logits, dim=1)
        
        class_loss = self.class_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        loss = class_loss

        return (loss, outputs)


class RobertaClassificationHead(nn.Module):

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(768 * 10 * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 3)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

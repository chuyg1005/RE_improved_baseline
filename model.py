import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.cuda.amp import autocast


class REModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.model_name, config=config)
        self.encoder.resize_token_embeddings(config.num_tokens)
        # self.encoder.gradient_checkpointing_enable()
        hidden_size = config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(hidden_size, args.num_class)
        )
        # self.lamb = float(os.environ.get("LAMBDA", 0.))
        mode = args.mode.split('@')
        self.mode = mode[0]
        self.kl_weight = 0.5
        self.lamb = 0.
        if len(mode) > 1:
            if self.mode in {'Debias'}:  # lamb
                self.lamb = float(mode[1])
            elif self.mode in {'RDrop', 'RDataAug'}:  # kl_weight
                self.kl_weight = float(mode[1])
            elif self.mode == 'MixDebias':
                self.kl_weight = float(mode[1])  # kl_weight
                self.lamb = float(mode[2])  # lamb
        # print(f'Using lamb = {self.lamb}')

    def compute_loss(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None):
        logits = self(input_ids, attention_mask, ss, os).float()
        if self.mode in {'default', 'EntityMask', 'DataAug', 'EntityOnly'}:
            loss = F.cross_entropy(logits, labels)
            return loss

        if self.mode in {"RDrop", "RDataAug"}:
            logits_p, logits_q = torch.chunk(logits, chunks=2, dim=0)
            labels_p, labels_q = torch.chunk(labels, chunks=2, dim=0)
            loss = F.cross_entropy(logits_p, labels_p)  # 正常计算损失
            regular_loss = (F.kl_div(F.log_softmax(logits_p, dim=-1), F.softmax(logits_q, dim=-1),
                                     reduction="batchmean")
                            + F.kl_div(F.log_softmax(logits_q, dim=-1), F.softmax(logits_p, dim=-1),
                                       reduction="batchmean")) * self.kl_weight
            loss += 1.0 * regular_loss
        elif self.mode == 'Focal':
            losses = F.cross_entropy(logits, labels, reduction='none')
            probs = F.softmax(logits, dim=-1)
            label_probs = torch.gather(probs, dim=1, index=labels.unsqueeze(1)).squeeze()
            weights = torch.pow(1 - label_probs, 2)
            loss = torch.dot(losses, weights) / labels.numel()
        elif self.mode == 'Reweight':
            losses = F.cross_entropy(logits, labels, reduction='none')
            probs = F.softmax(logits, dim=-1)
            label_probs = torch.gather(probs, dim=1, index=labels.unsqueeze(1)).squeeze()
            weights = 1 - label_probs
            loss = torch.dot(losses, weights) / labels.numel()
        elif self.mode == 'DFocal':
            logits_p, logits_eo = torch.chunk(logits, chunks=2, dim=0)
            labels_p, labels_eo = torch.chunk(labels, chunks=2, dim=0)
            probs_eo = F.softmax(logits_eo, dim=-1).detach()
            label_probs_eo = torch.gather(probs_eo, dim=1, index=labels_eo.unsqueeze(1)).squeeze()
            losses = F.cross_entropy(logits_p, labels_p, reduction='none')
            weights = torch.pow(1 - label_probs_eo, 2)
            loss = torch.dot(losses, weights) / labels_p.numel()
        elif self.mode == 'PoE':
            logits_p, logits_eo = torch.chunk(logits, chunks=2, dim=0)
            labels_p, labels_eo = torch.chunk(labels, chunks=2, dim=0)
            probs_p = F.softmax(logits_p, dim=-1)
            probs_eo = F.softmax(logits_eo, dim=-1).detach()
            probs_p = probs_p * probs_eo
            logits_p = torch.log(probs_p)
            loss = F.cross_entropy(logits_p, labels_p)
        elif self.mode == 'Debias':
            logits_p, logits_co = torch.chunk(logits, chunks=2, dim=0)
            labels_p, labels_co = torch.chunk(labels, chunks=2, dim=0)
            probs_p = F.softmax(logits_p, dim=-1)
            probs_co = F.softmax(logits_co, dim=-1).detach()
            label_probs_p = torch.gather(probs_p, dim=1, index=labels_p.unsqueeze(1)).squeeze()
            label_probs_co = torch.gather(probs_co, dim=1, index=labels_co.unsqueeze(1)).squeeze()
            biased_prob = label_probs_p - self.lamb * label_probs_co
            weights = torch.pow(1 - biased_prob, 2)
            losses = F.cross_entropy(logits_p, labels_p, reduction='none')
            loss = torch.dot(losses, weights) / labels_p.numel()
        elif self.mode == 'MixDebias':
            logits_p, logits_co, logits_q = torch.chunk(logits, chunks=3, dim=0)
            labels_p, labels_co, labels_q = torch.chunk(labels, chunks=3, dim=0)
            probs_p = F.softmax(logits_p, dim=-1)
            probs_co = F.softmax(logits_co, dim=-1).detach()
            label_probs_p = torch.gather(probs_p, dim=1, index=labels_p.unsqueeze(1)).squeeze()
            label_probs_co = torch.gather(probs_co, dim=1, index=labels_co.unsqueeze(1)).squeeze()
            probs = label_probs_p - self.lamb * label_probs_co
            weights = torch.pow(1 - probs, 2)
            losses = F.cross_entropy(logits_p, labels_p, reduction='none')
            loss = torch.dot(losses, weights) / labels_p.numel()
            regular_loss = (F.kl_div(F.log_softmax(logits_p, dim=-1), F.softmax(logits_q, dim=-1),
                                     reduction="batchmean")
                            + F.kl_div(F.log_softmax(logits_q, dim=-1), F.softmax(logits_p, dim=-1),
                                       reduction="batchmean")) * self.kl_weight
            loss += 1.0 * regular_loss

        else:
            assert 0

        return loss

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, ss=None, os=None):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)
        return logits

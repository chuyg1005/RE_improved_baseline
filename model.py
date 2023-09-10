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
        hidden_size = config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(hidden_size, args.num_class)
        )

    def compute_loss(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None):
        logits = self(input_ids, attention_mask, ss, os).float()
        if self.args.mode in {"RDrop", "RSwitch"}:
            loss = F.cross_entropy(logits, labels)
            logits_p, logits_q = torch.chunk(logits, chunks=2, dim=0)
            # labels_p, labels_q = torch.chunk(labels, chunks=2, dim=0)
            # loss = F.cross_entropy(logits_p, labels_p)
            regular_loss = (F.kl_div(F.log_softmax(logits_p, dim=-1),F.softmax(logits_q, dim=-1), reduction="batchmean")
                            + F.kl_div(F.log_softmax(logits_q, dim=-1), F.softmax(logits_p, dim=-1), reduction="batchmean")) * 0.5
            return loss + 1.0 * regular_loss
        elif self.args.mode == "DFocal":
            GAMMA = 2
            TEMPERATURE = 10
            logits_p, logits_q = torch.chunk(logits, chunks=2, dim=0)
            labels_p, labels_q = torch.chunk(labels, chunks=2, dim=0)
            # 计算概率
            probs_q = F.softmax(logits_q / TEMPERATURE, dim=-1)
            label_probs_q = torch.gather(probs_q, 1, labels_q.view(-1, 1)).view(-1).detach()

            # print(labels_p.numel())
            # 计算加权损失
            losses = F.cross_entropy(logits_p, labels_p, reduction="none")
            weights = torch.pow(1 - label_probs_q, GAMMA)
            loss = torch.dot(losses, weights)  / labels_p.numel()
            # loss = torch.log(loss) # 对损失进行自归一化
        else: # self.args.mode in {"default", "DataAug"}
            loss = F.cross_entropy(logits, labels)
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

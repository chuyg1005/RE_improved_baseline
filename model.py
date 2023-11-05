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

    def compute_loss(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None, power=None):
        logits = self(input_ids, attention_mask, ss, os).float()
        if self.args.mode in {"RDrop", "RSwitch"}:  # 使用RSwitch
            # loss = F.cross_entropy(logits, labels)
            logits_p, logits_q = torch.chunk(logits, chunks=2, dim=0)
            labels_p, labels_q = torch.chunk(labels, chunks=2, dim=0)
            loss = F.cross_entropy(logits_p, labels_p)  # 正常计算损失
            # loss1 = F.cross_entropy(logits_p, labels_p, reduction="none")  # 正常计算损失
            # loss2 = F.cross_entropy(logits_q, labels_q, reduction="none")  # 正常计算损失
            # regular_loss = F.mse_loss(loss1, loss2)
            regular_loss = (F.kl_div(F.log_softmax(logits_p, dim=-1), F.softmax(logits_q, dim=-1),
                                     reduction="batchmean")
                            + F.kl_div(F.log_softmax(logits_q, dim=-1), F.softmax(logits_p, dim=-1),
                                       reduction="batchmean")) * 0.5
            loss += 1.0 * regular_loss
        elif self.args.mode == 'MixDebias':
            logits_p, logits_q, logits_r = torch.chunk(logits, chunks=3, dim=0)
            labels_p, labels_q, labels_r = torch.chunk(labels, chunks=3, dim=0)
            # 加上Defocal
            probs_r = F.softmax(logits_r, dim=-1).detach()
            if power is not None:
                probs_r = torch.pow(probs_r, power)
                row_sums = torch.sum(probs_r, dim=1, keepdim=True)
                probs_r = probs_r / row_sums
            label_probs_r = torch.gather(probs_r, dim=1, index=labels_r.unsqueeze(1)).squeeze()
            losses = F.cross_entropy(logits_p, labels_p, reduction="none")
            weights = torch.pow(1 - label_probs_r, 2)
            loss = torch.dot(losses, weights) / labels_p.numel()
            regular_loss = (F.kl_div(F.log_softmax(logits_p, dim=-1), F.softmax(logits_q, dim=-1),
                                     reduction="batchmean")
                            + F.kl_div(F.log_softmax(logits_q, dim=-1), F.softmax(logits_p, dim=-1),
                                       reduction="batchmean")) * 0.5
            loss += 1.0 * regular_loss
        elif self.args.mode == 'PoE':
            logits_p, logits_q = torch.chunk(logits, chunks=2, dim=0)
            labels_p, labels_q = torch.chunk(labels, chunks=2, dim=0)
            probs_p = F.softmax(logits_p, dim=-1)
            probs_q = F.softmax(logits_q, dim=-1).detach()
            if power is not None:
                probs_q = torch.pow(probs_q, power)
                row_sums = torch.sum(probs_q, dim=1, keepdim=True)
                probs_q = probs_q / row_sums
            probs_q = probs_p * probs_q
            logits_p = torch.log(probs_q)
            loss = F.cross_entropy(logits_p, labels_p)

        elif self.args.mode in {"DFocal", "DataAugDFocal", "RSwitchDFocal"}:  # 数据增强和debiased focal一起使用
            GAMMA = 2
            TEMPERATURE = 1
            # p: 正常数据；q：entity-only数据
            logits_p, logits_q = torch.chunk(logits, chunks=2, dim=0)
            labels_p, labels_q = torch.chunk(labels, chunks=2, dim=0)
            # 计算概率
            probs_q = F.softmax(logits_q / TEMPERATURE, dim=-1)
            # 对概率进行退火
            if power is not None:
                # print(power)
                probs_q = torch.pow(probs_q, power)
                # 计算每行的总和
                row_sums = torch.sum(probs_q, dim=1, keepdim=True)

                # 将每行元素除以相应的行总和
                probs_q = probs_q / row_sums
            label_probs_q = torch.gather(probs_q, 1, labels_q.view(-1, 1)).view(-1).detach()

            # print(label_probs_q)
            # print(labels_p.numel())
            # 计算加权损失
            losses = F.cross_entropy(logits_p, labels_p, reduction="none")
            weights = torch.pow(1 - label_probs_q, GAMMA)
            loss = torch.dot(losses, weights) / labels_p.numel()
            # 增加正则化损失
            if self.args.mode == "RSwitchDFocal":
                logits_1, logits_2 = torch.chunk(logits_p, chunks=2, dim=0)
                regular_loss = (F.kl_div(F.log_softmax(logits_1, dim=-1), F.softmax(logits_2, dim=-1),
                                         reduction="batchmean")
                                + F.kl_div(F.log_softmax(logits_2, dim=-1), F.softmax(logits_1, dim=-1),
                                           reduction="batchmean")) * 0.5
                loss += 1.0 * regular_loss
        else:  # self.args.mode in {"default", "DataAug"}
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

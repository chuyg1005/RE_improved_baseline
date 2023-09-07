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
        logits = self(input_ids, attention_mask, ss, os)
        if self.args.mode == "default":
            loss = F.cross_entropy(logits.float(), labels)
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

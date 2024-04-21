import argparse
import json, os
# 使用TSNE进行降维
from sklearn.manifold import TSNE
import numpy as np
from transformers import AutoModel, AutoTokenizer
from utils import load4File, generate_md5_hash, default_collate_fn
import torch


class EmbeddingModel:
    def __init__(self, save_path, model_name='bert-base-cased', seed=42, batch_size=128, device='cuda'):
        self.save_path = save_path
        self.model_name = model_name
        self.device = device
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.encoder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = load4File(os.path.join(save_path, "processor.pkl"))
        self.seed = seed
        self.batch_size = batch_size

    def encode(self, data, reduction=False):
        # embeddings = self.model.encode(text, convert_to_tensor=convert_to_tensor).cpu().numpy()
        embeddings = []
        features = self.processor.encode(data, show_bar=True)
        num_batch = len(features) // self.batch_size + 1
        with torch.no_grad():
            for i in range(num_batch):
                batch_features = features[i * self.batch_size: (i + 1) * self.batch_size]
                input_ids, input_mask, labels, ss, _os = default_collate_fn(batch_features)
                outputs = self.encoder(
                    input_ids.to(self.device),
                    attention_mask=input_mask.to(self.device)
                )
                pooled_output = outputs[0]
                idx = torch.arange(input_ids.size(0)).to(input_ids.device)
                ss, _os = ss.to(self.device), _os.to(self.device)
                ss_emb = pooled_output[idx, ss]
                os_emb = pooled_output[idx, _os]
                h = torch.cat((ss_emb, os_emb), dim=-1)
                embeddings.append(h.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        if reduction:
            # # 使用TSNE进行降维
            tsne = TSNE(n_components=2, random_state=self.seed)
            embeddings = tsne.fit_transform(embeddings)
            # 使用pca进行降维
            # from sklearn.decomposition import PCA
            # pca = PCA(n_components=2)
            # embeddings = pca.fit_transform(embeddings)
        return embeddings


def load_data(dataset, k, name="test"):
    data = json.load(open(os.path.join('data', dataset, 'origin', name + '.json')))
    return data[:k]


def main(args):
    save_path = os.path.join('./ckpts', args.dataset, 'default', 'origin',
                             generate_md5_hash('typed_entity_marker_punct', 'origin', args.model_name, 'default',
                                               'train4debias', str(42)))
    model = EmbeddingModel(save_path, args.model_name, 42)
    # 输入文本数据，包括两种类型的文本

    # 合并文本数据
    test_data = load_data(args.dataset, args.k, "test")
    test_challenge_v1 = load_data(args.dataset, args.k, "test_challenge_v1")
    test_challenge_v2 = load_data(args.dataset, args.k, "test_challenge_v2")
    all_text_data = test_data + test_challenge_v1 + test_challenge_v2

    embeddings = model.encode(all_text_data, reduction=True)
    # embeddings = model.encode(all_text_data, convert_to_tensor=True, reduction=True)
    print(embeddings.shape)

    # save to file
    np.save(os.path.join('data', args.dataset, 'origin', 'embeddings.npy'), embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument("--dataset", type=str, choices=['tacred', 'tacrev', 'retacred'], default='tacred')
    parser.add_argument("--k", type=int, default=15000)
    args = parser.parse_args()
    main(args)

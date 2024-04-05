from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np


class StopsVectorGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def create_vector(self, stopwords: list):
        stopwords_embeddings = self._get_embedding_with_bert(stopwords)
        # stopwords_embeddings = self._generate_with_st(self.model_name)
        np.save('stops_vectors/768/ja.npy', stopwords_embeddings)

    def _get_embedding_with_bert(self, stopwords: list):
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertModel.from_pretrained(self.model_name)
        # ストップワードの埋め込みを計算
        stopwords_embeddings = {}
        for word in stopwords:
            # 文章をトークン化し、特殊トークンを追加
            tokens = tokenizer.encode(list(stopwords), add_special_tokens=True)
            # バッチの形状に整形
            input_ids = torch.tensor(tokens).unsqueeze(0)  # バッチサイズ1
            # BERTモデルに入力し、最終層の隠れ状態を取得
            with torch.no_grad():
                outputs = model(input_ids)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # 最終層の隠れ状態を平均化
            # Numpy配列に変換して埋め込みを取得
            embedding = embeddings.numpy()[0]  # バッチサイズ1なので最初の要素を取得
            stopwords_embeddings[word] = embedding
        return np.mean(list(stopwords_embeddings.values()), axis=0).reshape(1, 768)

    def _get_embedding_with_st(self, stopwords: list):
        stopwords_embeddings = {}
        model = SentenceTransformer(self.model_name)
        for word in stopwords:
            embedding = model.encode(word)
            stopwords_embeddings[word] = embedding
        return np.mean(list(stopwords_embeddings.values()), axis=0).reshape(1, 768)

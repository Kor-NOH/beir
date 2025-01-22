import json
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW


class BEIRDataset:
    def __init__(self, corpus_file: str, queries_file: str, qrels_file: str):
        # 코퍼스, 쿼리, Qrels 데이터를 로드합니다.
        with open(corpus_file, 'r', encoding='utf-8') as f:
            self.corpus = [json.loads(line) for line in f]

        with open(queries_file, 'r', encoding='utf-8') as f:
            self.queries = json.load(f)

        self.qrels = pd.read_csv(qrels_file, sep='\t', names=['query_id', 'corpus_id', 'score'])

    def get_training_data(self):
        # Qrels 기반으로 쿼리-문서 쌍 생성
        query_doc_pairs = []
        for _, row in self.qrels.iterrows():
            query = self.queries[row['query_id']]
            doc = next(item for item in self.corpus if item['_id'] == row['corpus_id'])
            query_doc_pairs.append((query, doc))
        return query_doc_pairs


# BEIR 데이터 로드
beir_data = BEIRDataset(
    corpus_file='KorQuAD-corpus.jsonl',
    queries_file='KorQuAD-queries.jsonl',
    qrels_file='qrels-train.tsv'
)

training_data = beir_data.get_training_data()

# 모델 초기화
dpr = DPR(model_path=("facebook/dpr-question_encoder-single-nq-base",
                      "facebook/dpr-ctx_encoder-single-nq-base"))


# 데이터 준비
def collate_fn(batch):
    queries, documents = zip(*batch)
    return queries, [{'title': doc.get('title', ''), 'text': doc['text']} for doc in documents]


train_loader = DataLoader(training_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

# 옵티마이저 설정
optimizer = AdamW(list(dpr.q_model.parameters()) + list(dpr.ctx_model.parameters()), lr=1e-5)

# 학습 루프
epochs = 3
for epoch in range(epochs):
    dpr.q_model.train()
    dpr.ctx_model.train()

    for batch in train_loader:
        queries, documents = batch

        # 쿼리와 문서 벡터 계산
        query_embeddings = dpr.encode_queries(queries)
        doc_embeddings = dpr.encode_corpus(documents)

        # 코사인 유사도를 계산하여 손실 함수 정의
        scores = torch.matmul(query_embeddings, doc_embeddings.T)
        labels = torch.arange(scores.size(0)).cuda()
        loss = torch.nn.CrossEntropyLoss()(scores, labels)

        # 역전파 및 파라미터 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# 학습 완료 후 저장
dpr.q_model.save_pretrained("output/question_encoder")
dpr.ctx_model.save_pretrained("output/context_encoder")

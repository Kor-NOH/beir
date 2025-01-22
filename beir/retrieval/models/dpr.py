from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm.autonotebook import trange
import torch
import json
import os

# Step 1: DPR Class Definition
class DPR:
    def __init__(self, model_path: tuple = None):
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path[0])
        self.q_model = DPRQuestionEncoder.from_pretrained(model_path[0])
        self.q_model.cuda()
        self.q_model.eval()

        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path[1])
        self.ctx_model = DPRContextEncoder.from_pretrained(model_path[1])
        self.ctx_model.cuda()
        self.ctx_model.eval()

    def encode_queries(self, queries: List[str], batch_size: int = 16) -> torch.Tensor:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                batch_queries = queries[start_idx:start_idx + batch_size]
                encoded = self.q_tokenizer(batch_queries, truncation=True, padding=True, return_tensors='pt')
                model_out = self.q_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                query_embeddings.extend(model_out.pooler_output.cpu())
        return torch.stack(query_embeddings)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8) -> torch.Tensor:
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size):
                batch = corpus[start_idx:start_idx + batch_size]
                titles = [doc['title'] for doc in batch]
                texts = [doc['text'] for doc in batch]
                encoded = self.ctx_tokenizer(titles, texts, truncation='longest_first', padding=True, return_tensors='pt')
                model_out = self.ctx_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                corpus_embeddings.extend(model_out.pooler_output.cpu())
        return torch.stack(corpus_embeddings)

# Step 2: Load BEIR Data
def load_beir_data(base_path: str):
    with open(os.path.join(base_path, "corpus.jsonl"), "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    with open(os.path.join(base_path, "queries.jsonl"), "r", encoding="utf-8") as f:
        queries = {line['id']: line['text'] for line in (json.loads(l) for l in f)}
    return corpus, queries

# Step 3: Train and Save Corpus Embeddings
def save_corpus_embeddings(dpr: DPR, corpus: List[Dict[str, str]], save_path: str):
    corpus_embeddings = dpr.encode_corpus(corpus)
    torch.save(corpus_embeddings, os.path.join(save_path, "corpus_embeddings.pt"))
    torch.save(corpus, os.path.join(save_path, "corpus_metadata.pt"))

# Step 4: Real-time Search with Trained Model
def real_time_search(dpr: DPR, save_path: str):
    corpus_embeddings = torch.load(os.path.join(save_path, "corpus_embeddings.pt"))
    corpus_metadata = torch.load(os.path.join(save_path, "corpus_metadata.pt"))

    while True:
        query = input("\n질문을 입력하세요 (종료하려면 'exit'): ")
        if query.lower() == "exit":
            print("종료합니다.")
            break

        query_embedding = dpr.encode_queries([query])
        scores = torch.matmul(query_embedding, corpus_embeddings.T)
        top_k = torch.topk(scores, k=5)

        print("\n검색 결과:")
        for idx in top_k.indices[0]:
            doc = corpus_metadata[idx]
            print(f"\n제목: {doc.get('title', '제목 없음')}")
            print(f"내용: {doc['text']}")

# Step 5: Main Execution
if __name__ == "__main__":
    base_path = "KorQuAD"  # Adjust to your data path
    save_path = "model_data"  # Directory to save embeddings
    os.makedirs(save_path, exist_ok=True)

    # Load Data
    print("데이터 로드 중...")
    corpus, queries = load_beir_data(base_path)

    # Initialize DPR
    print("모델 초기화 중...")
    dpr = DPR(model_path=("facebook/dpr-question_encoder-single-nq-base",
                          "facebook/dpr-ctx_encoder-single-nq-base"))

    # Save Corpus Embeddings
    print("문서 임베딩 생성 및 저장 중...")
    save_corpus_embeddings(dpr, corpus, save_path)

    # Start Real-Time Search
    print("실시간 검색 준비 완료!")
    real_time_search(dpr, save_path)

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch

class DPR:
    def __init__(self, model_path: Union[str, Tuple] = None, **kwargs):
        # DPR 모델과 토크나이저를 로드하고 설정

        # 쿼리 인코더의 토크나이저 및 모델 초기화
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path[0])
        self.q_model = DPRQuestionEncoder.from_pretrained(model_path[0])
        self.q_model.cuda()
        self.q_model.eval()

        # 컨텍스트 인코더의 토크나이저 및 모델 초기화
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path[1])
        self.ctx_model = DPRContextEncoder.from_pretrained(model_path[1])
        self.ctx_model.cuda()
        self.ctx_model.eval()

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> torch.Tensor:
        # 쿼리를 벡터로 인코딩

        query_embeddings = []   # 결과를 저장할 리스트
        with torch.no_grad():   # 파라미터 업데이터 비활성화
            for start_idx in trange(0, len(queries), batch_size):   # 배치 단위로 처리
                # 현재 배치의 질문을 토큰화
                encoded = self.q_tokenizer(queries[start_idx:start_idx+batch_size], truncation=True, padding=True, return_tensors='pt')

                # 토큰화된 질문을 모델에 입력해 벡터 계산
                model_out = self.q_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                query_embeddings += model_out.pooler_output # 결과를 리스트에 추가

        return torch.stack(query_embeddings)    # 리스트를 텐서로 변환해 반환

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> torch.Tensor:
        # 문서(corpus)를 벡터로 인코딩

        corpus_embeddings = []  # 결과를 저장할 리스트
        with torch.no_grad():   # 파라미터 업데이트 비활성화
            for start_idx in trange(0, len(corpus), batch_size):    # 배치 단위로 처리
                # 현재 배치의 제목과 본문 추출
                titles = [row['title'] for row in corpus[start_idx:start_idx+batch_size]]
                texts = [row['text']  for row in corpus[start_idx:start_idx+batch_size]]

                # 제목과 본문을 함께 토큰화
                encoded = self.ctx_tokenizer(titles, texts, truncation='longest_first', padding=True, return_tensors='pt')

                # 토큰화된 문서를 모델에 입력해 벡터 계산
                model_out = self.ctx_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                corpus_embeddings += model_out.pooler_output.detach()   # 결과를 리스트에 추가

        return torch.stack(corpus_embeddings)   # 리스트를 텐서로 변환해 반환
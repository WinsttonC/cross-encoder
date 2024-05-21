import wandb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import chromadb
from joblib import load
from sentence_transformers import SentenceTransformer
# model_name='intfloat/multilingual-e5-large'
# model = SentenceTransformer(model_name)
# model.encode(sentences=['texts'], device='mps')
import random
from tqdm import tqdm
import numpy as np



class CreateExperiment:
    def __init__(self,
                 df,
                 model_name,
                 model,
                 tokens_per_chunk=510,
                 n_chunks=5
                 ):

        self.tokens_per_chunk = tokens_per_chunk
        self.model_name = model_name
        self.df = df
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name)
        self.embedding_function_2 = SentenceTransformer(model_name)
        self.model = load(model)
        self.n_chunks = n_chunks
        self.chunks = self.text_token_splitter("\n\n".join(df['texts']))
        self.chroma_collection = self.create_chroma()

    def text_token_splitter(self, texts):
        token_splitter = SentenceTransformersTokenTextSplitter(model_name=self.model_name,
                                                               chunk_overlap=0,
                                                               tokens_per_chunk=self.tokens_per_chunk)
        token_spliter_chunks = token_splitter.split_text(texts)

        return token_spliter_chunks

    def create_chroma(self):
        chroma_client = chromadb.Client()

        random_float = round(random.random(), 4)
        name = f'test{random_float}fc'
        chroma_collection = chroma_client.create_collection(
            name, embedding_function=self.embedding_function)
        ids = [str(i) for i in range(len(self.chunks))]
        chroma_collection.add(ids=ids, documents=self.chunks)
        chroma_collection.count()
        return chroma_collection

    def get_rel_chunks(self, query):
        """
        args:
            query: str - вопрос к базе знаний, по которому в ищутся релевантные чанки
            n_chunks: int - сколько релевантных чанков возвращать
        returns:
            output_lst: list - список релевантных чанков
        """
        results = self.chroma_collection.query(
            query_texts=[query], n_results=self.n_chunks)
        retrieved_documents = results['documents'][0]
        output_lst = []

        for document in retrieved_documents:
            output_lst.append(document)

        return output_lst

    def longest_common_substring(self, str1, str2):
        """

        args:
            str1: str
            str2: str
        """
        table = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
        longest = 0
        lcs = ""

        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                    if table[i][j] > longest:
                        longest = table[i][j]
                        lcs = str1[i - longest:i]
                else:
                    table[i][j] = 0
        return len(lcs.split(' '))

    def mrr(self, rerank=False, n=4):
        """

        args:
            data: pd.DataFrame - датафрейм с вопросами и ответами
            n: int - какую часть от ответа считать достаточной при определении релевантности чанка
        """
        mrr_metrics = []

        questions = self.df['question'].to_list()
        for question in questions:
            if rerank:
                relevant_docs = self.get_rel_chunks(question)
                relevant_docs = self.rerank(question, relevant_docs)
            else:
                relevant_docs = self.get_rel_chunks(question)

            pos = 1
            for doc in relevant_docs:
                answer = self.df.loc[self.df['question']
                                     == question, 'answer'].values[0]
                if answer in doc:
                    mrr_metrics.append(1/pos)
                    break
                elif self.longest_common_substring(answer, doc) >= len(answer.split(" ")) // n:
                    mrr_metrics.append(1/pos)
                    break

                pos += 1

        while len(mrr_metrics) < len(questions):
            mrr_metrics.append(0)
        mean_mrr = sum(mrr_metrics)/len(mrr_metrics)
        return mean_mrr

    def rerank(self, query, preranked_docs):
        data_dict = {}
        for doc in preranked_docs:
            _txt = query + ' // ' + doc
            txt_emb = self.embedding_function_2.encode(
                sentences=[_txt], device='mps')  # -> arr[]
            score = self.model.predict_proba(txt_emb)
            data_dict[doc] = score[0][1]
        reranked_documents = [el[0] for el in sorted(
            data_dict.items(), key=lambda item: item[1], reverse=True)]

        return reranked_documents

    def create_experiment_report(self):
        print(
            f'Experiment report:\nModel name: {self.model_name} | Tokens per chunk: {self.tokens_per_chunk}\nNumber of chunks: todo\nMean MRR for documents {self.mrr()}')


df = pd.read_csv('data/test_data_with_questions_answ_2.csv')
df.rename(columns={'Unnamed: 3': 'answer'}, inplace=True)
df = df[df['answer'].notnull()]


chunks_size_by_tokens = np.array([128, 256, 510, 1024, 2048])
ml_algs = ['log_reg'] 

log_reg_models = {
    "intfloat/multilingual-e5-large": "ml_models_weights/logistic_regression_model_1024.joblib",
    "cointegrated/LaBSE-en-ru": "ml_models_weights/logistic_regression_model_768.joblib",
    "cointegrated/rubert-tiny2": "ml_models_weights/logistic_regression_model_312.joblib",
}

df_logs = pd.read_csv('./data/logs.csv')

for ml_alg in ml_algs:
    for lm_model in tqdm(log_reg_models, desc='ML Algorithm'):
        current_logreg = log_reg_models[lm_model]
        config = {
            'ML_algorithm': current_logreg,
            'embedding_model': lm_model,
        }
        wandb.init(project="cross-encoder",
                   job_type="LogReg testing",
                   name=lm_model,
                   config=config,
                   save_code=True)

        for chunk_size in tqdm(chunks_size_by_tokens, desc='chunks_size_by_tokens'):

            if chunk_size > 510:  # if chunk is more than
                if lm_model == 'intfloat/multilingual-e5-large':
                    lm_model = 'local_lang_models/LSGC_4096_intfloat_multilingual_e5_large'

                if lm_model == 'cointegrated/LaBSE-en-ru':
                    lm_model = 'local_lang_models/LSGC_4096_cointegrated_LaBSE_en_ru'

            lr_expr = CreateExperiment(df=df, model_name=lm_model,
                                       model=current_logreg,
                                       tokens_per_chunk=chunk_size)            
            mrr_basic = lr_expr.mrr(rerank=False)
            mrr_with_classification = lr_expr.mrr(rerank=True)
            
            logs_df = pd.DataFrame({
                'chunk_size': [chunk_size], 
                'ml_model': [current_logreg], 
                'emb_model': [lm_model], 
                'mrr_basic': [mrr_basic], 
                'mrr_cls': [mrr_with_classification]})

            df_logs = pd.concat([df_logs, logs_df], ignore_index=True)
            df_logs.to_csv('./data/logs.csv', index=False)
            
            chunk_size_for_wb = chunk_size

            wandb.log({'chunk_size': chunk_size_for_wb, 'MRR_basic': mrr_basic,
                      'MRR_with_clf': mrr_with_classification})
            print("ml_algs: log_reg")
            print(f"lm_model : {lm_model}")
            print(f"chunk_size : {chunk_size}")
            print(f"MRR_SCORE: {mrr_with_classification}")
            print("\n")
            # without
            print("ml_algs: BASIC")
            print(f"lm_model : {lm_model}")
            print(f"chunk_size : {chunk_size}")
            print(f"MRR_SCORE: {mrr_basic}")
            print("\n")
            
        wandb.finish()

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import chromadb

# TODO: добавить метод вызова моделей классификации
# TODO: добавить количество чанков в векторном хранилище в report

class CreateExperiment:
    def __init__(self,
                 df,
                 model_name,
                 tokens_per_chunk=510,
                 n_chunks=5
                 ):
        
        self.tokens_per_chunk = tokens_per_chunk
        self.model_name = model_name
        self.df = df
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
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
        chroma_collection = chroma_client.create_collection("test", embedding_function=self.embedding_function)
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
        results = self.chroma_collection.query(query_texts=[query], n_results=self.n_chunks)
        retrieved_documents = results['documents'][0]
        output_lst = []

        for document in retrieved_documents:
            output_lst.append(document)
            
        return output_lst 
    
    def longest_common_substring(str1, str2):
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

    def mrr(self, n=4):
        """

        args:
            data: pd.DataFrame - датафрейм с вопросами и ответами
            n: int - какую часть от ответа считать достаточной при определении релевантности чанка
        """
        mrr_metrics = []

        questions = self.df['question'].to_list()
        for question in questions:
            relevant_docs = self.get_rel_chunks(question)

            pos = 1
            for doc in relevant_docs:
                answer = self.df.loc[self.df['question']==question, 'answer'].values[0]
                if answer in doc:
                    mrr_metrics.append(1/pos)
                    break
                elif self.longest_common_substring(answer, doc) >= len(answer.split(" ")) // n:
                    mrr_metrics.append(1/pos)
                    break

                pos += 1
            
            log_question_counter += 1

        while len(mrr_metrics) < len(questions):
            mrr_metrics.append(0)
        mean_mrr = sum(mrr_metrics)/len(mrr_metrics)
        return mean_mrr
        # return f'Mean MRR for documents {sum(mrr_metrics)/len(mrr_metrics)}'

    def create_experiment_report(self):
        print(f'Experiment report:\nModel name: {self.model_name} | Tokens per chunk: {self.tokens_per_chunk}\nNumber of chunks: todo\nMean MRR for documents {self.mrr()}')


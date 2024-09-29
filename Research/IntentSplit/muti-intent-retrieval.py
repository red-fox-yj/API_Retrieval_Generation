import json
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ijson
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# 添加结束标记的函数
def add_eos(input_examples, eos_token):
    return [input_example + eos_token for input_example in input_examples]

def load_qa_from_jsonfile(json_file):
    """使用 ijson 库从多个 JSON 文件中流式加载 Q 和 A 对，并将 `name` 附加到每个 QA 对象中"""
    print("Loading QA pairs from JSON file list...")
    questions = []
    qa_pairs = []

    # 遍历每个 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for qa in data:
            questions.append(qa["Q"])  # 提取问题并存储
            qa_pairs.append(qa)  # 存储整个 QA 对象 
                
    print(f"Loaded {len(qa_pairs)} QA pairs from {json_file} .")
    return qa_pairs, questions


def create_faiss_index(embeddings):
    """创建 Faiss 索引"""
    print(f"Creating Faiss index with {embeddings.shape[0]} embeddings...")
    dimension = embeddings.shape[1]  # 嵌入向量的维度
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("Faiss index created.")
    return index

def search_similar_qa(embedding_test, index, test_qas, k=3, dedup=True):
    """检索与输入查询最相似的 QA 对，支持去重"""
    unique_results = []
    seen_questions = set()
    offset = 0
    
    # 循环，直到获得 k 个唯一的结果
    while len(unique_results) < k:
        distances, indices = index.search(embedding_test, k + offset)
        results = [test_qas[idx] for idx in indices[0]]
        
        # 去重：只添加还未见过的 `Q`
        for result in results:
            if dedup and result['Q'] not in seen_questions:
                seen_questions.add(result['Q'])
                unique_results.append(result)
            elif not dedup:
                unique_results.append(result)
                
            if len(unique_results) >= k:
                break

        # 如果未找到足够的唯一结果，增加偏移量，继续检索更多
        offset += k

    return unique_results[:k]


def evaluate_model(train_qas,test_qas,test_embeddings,index, k):
    """评估模型的准确率和精确率，精确率不区分去重场景"""
    print(f"Evaluating model with top-k {k}...")

    correct_retrieved_no_dedup = defaultdict(int)
    correct_retrieved_dedup = defaultdict(int)
    precision_per_query = [] if k > 1 else None  # 如果 k = 1，精确率不需要计算
    precision_per_name_query = defaultdict(list) if k > 1 else None
    total_questions_per_name = defaultdict(int)

    print(f"Running evaluation on {len(test_qas)} test queries...")
    
    for test_qa, test_embedding in zip(test_qas, test_embeddings):
        current_name = test_qa['name']
        total_questions_per_name[current_name] += 1

        test_embedding_np = np.array(test_embedding).reshape(1, -1)
        top_k_results_no_dedup = search_similar_qa(test_embedding_np, index, train_qas, k, dedup=False)
        top_k_results_dedup = search_similar_qa(test_embedding_np, index, train_qas, k, dedup=True)

        if any(result['name'] == current_name for result in top_k_results_no_dedup):
            correct_retrieved_no_dedup[current_name] += 1
        if any(result['name'] == current_name for result in top_k_results_dedup):
            correct_retrieved_dedup[current_name] += 1

        if k > 1:
            precision = sum(result['name'] == current_name for result in top_k_results_no_dedup) / min(len(top_k_results_no_dedup), k) if top_k_results_no_dedup else 0
            precision_per_query.append(precision)
            precision_per_name_query[current_name].append(precision)

    total_questions = len(test_qas)
    overall_accuracy_no_dedup = sum(correct_retrieved_no_dedup.values()) / total_questions
    overall_accuracy_dedup = sum(correct_retrieved_dedup.values()) / total_questions

    if k > 1:
        overall_precision = np.mean(precision_per_query)
    else:
        overall_precision = None

    print("Evaluation completed.")
    return overall_accuracy_no_dedup, overall_accuracy_dedup, overall_precision


def save_results_to_json(results, json_file):
    """保存评估结果到相对路径的 JSON 文件"""
    # 处理路径，确保父目录存在
    dir_name = os.path.dirname(json_file)
    
    # 如果目录存在并且不是当前目录，创建目录
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    print(f"Saving results to {json_file}...")
    
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    
    print("Results saved.")


# nvidia/NV-Embed-v2
class ModelHandlerV1:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model.max_seq_length = 32768
        model.tokenizer.padding_side = "right"
        print(f"Model {model_name} loaded.")
        return model.to('cuda')

    def get_batch_embeddings(self, texts, batch_size=32, prefix=None):
        all_embeddings = []
        if prefix:
            texts = [prefix + text for text in texts]
        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(batch_texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
            all_embeddings.append(embeddings.cpu().numpy())
        print("Embeddings generated.")
        return np.vstack(all_embeddings)


def main(train_file,test_file,model_name, ks=[3], test_sizes=[0.2]):
    # 根据模型名称选择相应的模型处理器
    model_handler = ModelHandlerV1(model_name)
    
    train_qas, train_qs = load_qa_from_jsonfile(train_file)
    train_embeddings = model_handler.get_batch_embeddings(train_qs)

    test_qas, test_qs = load_qa_from_jsonfile(test_file)
    test_embeddings = model_handler.get_batch_embeddings(test_qs)

    all_results = {}

    index = create_faiss_index(train_embeddings)
    for k in ks:
        print(f"Evaluating for k={k}...")
        overall_accuracy_no_dedup, overall_accuracy_dedup, overall_precision = evaluate_model(
            train_qas,test_qas,test_embeddings,index, k)
        
        print(f"Results for k={k}:")
        print(f"  Overall accuracy_no_dedup: {overall_accuracy_no_dedup:.4f}")
        print(f"  Overall accuracy_dedup: {overall_accuracy_dedup:.4f}")
        if k > 1:
            print(f"  Overall precision: {overall_precision:.4f}")

        all_results[f"k_{k}"] = {
            "overall_accuracy_no_dedup": overall_accuracy_no_dedup,
            "overall_accuracy_dedup": overall_accuracy_dedup,
            "overall_precision": overall_precision
        }

    save_results_to_json(all_results, f"result/muti-intent-retrieval/{model_name.replace('/', '_')}_k_{k}.json")

if __name__ == "__main__":
    train_file = 'data/qa.json'
    test_file = 'results/intent_split_Qwen2.5-7B-Instruct.json'
    ks = [1, 3, 5]
    main( train_file = train_file,test_file=test_file,model_name='nvidia/NV-Embed-v2', ks=ks)

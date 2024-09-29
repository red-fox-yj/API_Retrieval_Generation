import json
import os
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

# 初始化 Qwen 模型路径
qwen_7b_model_name = "Qwen/Qwen2.5-7B-Instruct"
qwen_72b_model_name = "/remote-home1/share/models/Qwen2-72B-Instruct"

# 初始化 DeepSeek API
openai.api_key = "sk-162e7df7b44f4f2dab3d03ae28b3bbd1"
deepseek_url = "https://api.deepseek.com"

# 加载 Qwen 模型和 tokenizer
def load_qwen_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

qwen_7b_model, qwen_7b_tokenizer = load_qwen_model(qwen_7b_model_name)
qwen_72b_model, qwen_72b_tokenizer = load_qwen_model(qwen_72b_model_name)

# 构建 prompt 让模型判断意图拆解的准确性并打分
def create_prompt_for_scoring(original_query, decomposed_queries):
    decomposed_str = '\n'.join([f"{i+1}. {dq}" for i, dq in enumerate(decomposed_queries)])
    prompt = f"""原始请求: {original_query}
模型拆解的意图:
{decomposed_str}

请根据以下两个标准对意图拆解进行评分：
1. 意图拆解的数量是否与原始请求中的意图数量一致（1 分）。
2. 如果数量一致，判断拆解的意图是否与原始请求的意图一致（2 分）。

请严格按照以下格式给出最终的分数和简要说明：
分数：X 分
简要说明：X 的理由

示例：
分数：3 分
简要说明：意图拆解数量与内容完全一致。
"""
    return prompt

# 使用 Qwen 模型生成评分
def get_qwen_model_score(prompt, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant tasked with evaluating the accuracy of intent decomposition. Please follow the provided instructions carefully."},
        {"role": "user", "content": prompt}
    ]
    # 构建输入文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成模型响应
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    # 解码生成的内容
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# 使用 DeepSeek V2 生成评分
def get_deepseek_model_score(prompt):
    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant tasked with evaluating the accuracy of intent decomposition. Please follow the provided instructions carefully."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# 提取模型响应中的评分
def extract_score_from_response(response):
    match = re.search(r"分数：(\d) 分", response)
    if match:
        score = int(match.group(1))
    else:
        score = 0  # 如果未找到分数，返回 0
    return score

# 评估函数并保存结果到 JSON 文件
def evaluate_intent_accuracy(data, output_json_file, model_type, model=None, tokenizer=None):
    total_queries = len(data)
    success_count = 0
    results = []

    for item in tqdm(data, desc=f"Evaluating Intent Accuracy ({model_type})"):
        original_query = item["original_query"]
        decomposed_queries = item["decomposed_queries"]

        # 构建 prompt
        prompt = create_prompt_for_scoring(original_query, decomposed_queries)
        
        # 根据模型类型生成评分
        if model_type == "Qwen-2.5-7B":
            generated_response = get_qwen_model_score(prompt, qwen_7b_model, qwen_7b_tokenizer)
        elif model_type == "Qwen-2.5-72B":
            generated_response = get_qwen_model_score(prompt, qwen_72b_model, qwen_72b_tokenizer)
        elif model_type == "DeepSeek V2":
            generated_response = get_deepseek_model_score(prompt)
        
        # 提取评分
        score = extract_score_from_response(generated_response)
        
        if score == 3:
            success_count += 1

        # 保存每次的结果
        result = {
            "original_query": original_query,
            "decomposed_queries": decomposed_queries,
            "prompt": prompt,
            "model_response": generated_response,
            "score": score
        }
        results.append(result)
    
    # 计算成功率
    success_rate = success_count / total_queries if total_queries > 0 else 0
    print(f"Success rate ({model_type}): {success_rate:.2f}")

    # 保存结果和成功率到 JSON 文件
    output_data = {
        "success_rate": success_rate,
        "results": results
    }

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

# 读取 JSON 文件并调用评估函数
def run_evaluation(json_file_path):
    # 使用 os.path.splitext 拆分文件名和扩展名，并构造新的文件名
    base_name, _ = os.path.splitext(json_file_path)

    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 评估 Qwen-2.5-7B
    output_json_file_qwen_7b = f'result/{base_name}_qwen_7b_evaluation.json'
    evaluate_intent_accuracy(data, output_json_file_qwen_7b, model_type="Qwen-2.5-7B")

    # 评估 Qwen-2.5-72B
    output_json_file_qwen_72b = f'result/{base_name}_qwen_72b_evaluation.json'
    evaluate_intent_accuracy(data, output_json_file_qwen_72b, model_type="Qwen-2.5-72B")

    # 评估 DeepSeek V2
    output_json_file_deepseek = f'result/{base_name}_deepseek_evaluation.json'
    evaluate_intent_accuracy(data, output_json_file_deepseek, model_type="DeepSeek V2")

# 调用评估函数
json_file_path = 'results/intent_split_Qwen2.5-7B-Instruct.json'  # 替换为实际文件路径
run_evaluation(json_file_path)

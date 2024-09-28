import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

# 加载 Qwen2.5-72B-Instruct 模型和 tokenizer
model_path = "/remote-home1/share/models/Qwen2-72B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 构建 prompt 让 Qwen 判断意图拆解的准确性，并打分
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

# 使用模型生成评分
def get_model_score(prompt):
    # 指定instruction，通过system消息明确传递给模型
    messages = [
        {"role": "system", "content": "You are a helpful assistant tasked with evaluating the accuracy of intent decomposition. Please follow the provided instructions carefully. Your task is to score based on the number of intents and their correctness."},
        {"role": "user", "content": prompt}
    ]
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

# 从模型响应中提取评分
def extract_score_from_response(response):
    # 使用正则表达式提取类似 "分数：X 分" 的格式
    match = re.search(r"分数：(\d) 分", response)
    if match:
        score = int(match.group(1))  # 提取并转换为整数
    else:
        score = 0  # 如果没有匹配到，默认返回 0
    return score

# 评估函数并保存结果到 json 文件
def evaluate_intent_accuracy(data, output_json_file):
    total_queries = len(data)
    total_score = 0  # 用于计算总分
    results = []  # 存储每次的评分结果

    for item in tqdm(data, desc="Evaluating Intent Accuracy"):
        original_query = item["original_query"]
        decomposed_queries = item["decomposed_queries"]

        # 构建 prompt
        prompt = create_prompt_for_scoring(original_query, decomposed_queries)
        
        # 使用模型生成评分
        generated_response = get_model_score(prompt)
        
        # 提取模型给出的评分
        score = extract_score_from_response(generated_response)
        
        total_score += score

        # 保存每次的结果和得分
        result = {
            "original_query": original_query,
            "decomposed_queries": decomposed_queries,
            "prompt": prompt,
            "model_response": generated_response,
            "score": score
        }
        results.append(result)
    
    # 计算平均分
    average_score = total_score / total_queries if total_queries > 0 else 0
    print(f"Average Score: {average_score:.2f}")

    # 保存结果和平均分到 JSON 文件
    output_data = {
        "average_score": average_score,
        "results": results
    }

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    return average_score

# 读取 JSON 文件并调用评估函数
json_file_path = 'results/output_Qwen2.5-7B-Instruct.json'
output_json_file = f'{json_file_path}_evaluation.json'

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 调用评估函数并保存结果
average_score = evaluate_intent_accuracy(data, output_json_file)
print(f"Results saved to {output_json_file}")

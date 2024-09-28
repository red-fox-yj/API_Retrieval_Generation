import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 定义函数，用于发送指令给模型
def parse_intent_with_qwen(query, model, tokenizer):
    # 使用system角色来描述任务和要求
    messages = [
        {"role": "system", "content": """
You are Qwen, an AI assistant created by Alibaba Cloud. Your task is to help users decompose complex queries. 
If a query contains multiple intents, split them into individual intents and number them clearly. 
If the query has only one intent, mark it as '单意图' (single intent).

For complex queries that involve multiple actions or goals, ensure each distinct task is listed separately. Use the following format for your response:

- For multiple intents, return:
多意图
1. [Intent 1]
2. [Intent 2]
...

- For a single intent, return:
单意图
[Intent]

Please strictly follow this format and ensure clarity in the output.

### Examples:

Example 1:
Q: 播放最近听的歌曲后，播放我收藏的《中文流行》歌单
A: 
多意图
1. 播放最近听的歌曲
2. 播放我收藏的《中文流行》歌单

---

Example 2:
Q: 导航显示还有多少时间到达?
A: 
单意图
查询导航显示的剩余时间到达目的地

---

Example 3:
Q: 播放我最近听的音乐，同时打开导航并告诉我剩余时间。
A: 
多意图
1. 播放我最近听的音乐
2. 打开导航并查询剩余时间

---

Now, analyze the following query and provide a structured response:
"""},

        # 传递用户查询
        {"role": "user", "content": f"Q: {query}\nA:"}
    ]

    # 应用聊天模板并生成输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 使用模型生成输出
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # 解码生成的文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 提取意图
    parsed_intents = []
    if "多意图" in response:
        response_lines = response.splitlines()
        for line in response_lines:
            # 通过判断是否有编号（如"1."）开头来确定每个意图
            if line.strip().startswith(tuple(str(i) + '.' for i in range(1, 10))):
                parsed_intents.append(line.strip()[3:].strip())  # 去掉编号，保留意图内

    elif "单意图" in response:
        # 处理单意图的情况
        result_start = response.find("单意图") + len("单意图")
        parsed_intents.append(response[result_start:].strip())
    else:
        print("无法识别的意图格式")
    
    return parsed_intents, response

# 从文件读取原始数据
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 保存数据到新的JSON文件
def save_json_file(data, output_file):
    dir_name = os.path.dirname(output_file)
    
    # 如果目录不存在，创建目录
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 处理意图拆解
def process_intents(input_file, output_file, model_name):
    # 读取原始数据
    data = read_json_file(input_file)

    # 加载指定的模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # 用于存储拆解后的结果
    processed_data = []

    # 使用 tqdm 进度条包裹对 data 的遍历
    for item in tqdm(data[360:], desc="Processing items", unit="item"):
        questions = item.get("question", [])
        name_fields = item.get("name", [])

        # 对每个 question 进行处理
        for question in questions:
            # 利用 Qwen 模型拆解意图
            parsed_intents, response = parse_intent_with_qwen(question, model, tokenizer)
            
            # 构建拆解后的结果
            result = {
                "original_query": question,
                "decomposed_queries": parsed_intents,  # 将返回的多个意图按换行拆分
                "name_fields": name_fields,
                "response": response
            }

            # 将结果添加到 processed_data 列表中
            processed_data.append(result)
    
    # 保存拆解后的数据
    save_json_file(processed_data, output_file)

# 定义测试函数，用于不同模型测试
def test_models(input_file):
    models = {
        "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        # 可以在这里添加更多模型，例如 Qwen-14B 或 Qwen-72B
        # "Qwen-2.5-14B": "Qwen/Qwen-2.5-14B",
        # "Qwen-2.5-72B": "Qwen/Qwen-2.5-72B"
    }
    
    for model_name, model_path in models.items():
        output_file = f'results/output_{model_name}.json'
        print(f"Processing with {model_name}...")
        process_intents(input_file, output_file, model_path)
        print(f"Results saved to {output_file}")

# 调用测试函数
input_file = 'data/多意图数据(部分9.25).json'  # 输入文件路径
test_models(input_file)

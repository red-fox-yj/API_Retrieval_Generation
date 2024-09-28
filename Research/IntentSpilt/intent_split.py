import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 从 JSON 文件中读取 API 信息
def read_api_info(file_list):
    """
    从多个 JSON 文件中读取所有 API 名称和其描述。
    :param file_list: 包含 JSON 文件路径的列表
    :return: 包含所有 API 名称和描述的字典
    """
    api_info = {}
    
    for file_path in file_list:
        if os.path.exists(file_path) and file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    api_data = json.load(f)
                    # 遍历每个文件中的 API 数据
                    for api in api_data:
                        api_name = api.get('name')
                        api_description = api.get('description', 'No description available')
                        if api_name:
                            api_info[api_name] = api_description
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    return api_info

# 定义标准 prompt 解析函数 (方案 1)
def parse_intent_with_standard_qwen(query, model, tokenizer):
    """
    使用标准 prompt 解析意图，不包含 API 信息。
    :param query: 用户输入的查询
    :param model: 模型对象
    :param tokenizer: 分词器对象
    """
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
                parsed_intents.append(line.strip()[3:].strip())  # 去掉编号，保留意图内容

    elif "单意图" in response:
        # 处理单意图的情况
        result_start = response.find("单意图") + len("单意图")
        parsed_intents.append(response[result_start:].strip())
    else:
        print("无法识别的意图格式")
    
    return parsed_intents, response

# 定义带有 API 信息的 prompt 解析函数 (方案 2)
def parse_intent_with_api_qwen(query, model, tokenizer, api_info, api_names):
    """
    使用第二种带有 API 信息的 prompt 解析意图，并将 API 信息嵌入到用户查询中。
    :param query: 用户输入的查询
    :param model: 模型对象
    :param tokenizer: 分词器对象
    :param api_info: 包含 API 名称和描述的字典
    :param api_names: 该查询所对应的 API 名称列表
    """
    # 根据 API 名称从 api_info 中找到对应描述
    api_descriptions = [f"- `{api_name}`: {api_info.get(api_name, 'No description available')}" for api_name in api_names]
    
    # 将 API 描述组合成字符串
    user_api_info_str = "\n".join(api_descriptions)
    
    messages = [
        {"role": "system", "content": f"""
You are Qwen, an AI assistant created by Alibaba Cloud. Your task is to decompose complex user queries into individual intents. You have access to relevant API information that can help you better understand and infer user intents. Each query will correspond to one or more API calls, and you should use your knowledge of the available APIs to guide your intent decomposition.

### Task Instructions:
1. **Understand the user query** and use the provided API information to infer the user's intent as accurately as possible.
2. If the query contains multiple intents, decompose them into individual, distinct tasks, each corresponding to a single API call.
3. If the query contains only one intent, label it as '单意图' (single intent).
4. **Use the API descriptions** to help better understand and split user queries into actionable, atomic intents.

### Examples:

Example 1:
API Information:
- `music.play_recent`: This API plays the most recent songs the user has listened to.
- `playlist.play_collection`: This API plays a specific playlist that the user has saved.

Q: 播放最近听的歌曲后，播放我收藏的《中文流行》歌单
A: 
多意图
1. 播放最近听的歌曲
2. 播放我收藏的《中文流行》歌单

---

Example 2:
API Information:
- `navigation.query_remaining_time`: This API retrieves the remaining time to the destination.

Q: 导航显示还有多少时间到达?
A: 
单意图
查询导航显示的剩余时间到达目的地

---

Example 3:
API Information:
- `music.play_recent`: This API plays the most recent songs the user has listened to.
- `navigation.query_remaining_time`: This API retrieves the remaining time to the destination.

Q: 播放我最近听的音乐，同时打开导航并告诉我剩余时间。
A: 
多意图
1. 播放我最近听的音乐
2. 打开导航并查询剩余时间

---

Now, analyze the following query and provide a structured response, using the API information provided with the query to infer and decompose the intents if necessary.
"""},

        # 传递用户查询并嵌入对应的 API 信息
        {"role": "user", "content": f"Q: {query}\n### API Information:\n{user_api_info_str}\nA:"}
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
                parsed_intents.append(line.strip()[3:].strip())  # 去掉编号，保留意图内容

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
def process_intents(input_file, output_file, model_name, api_info=None, prompt_type="standard"):
    """
    处理意图拆解，支持标准 prompt 和带有 API 信息的 prompt
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param model_name: 使用的模型
    :param api_info: 如果使用 API 信息，传递 API 描述
    :param prompt_type: 选择 "standard" 或 "api" 类型的 prompt
    """
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
    for item in tqdm(data, desc="Processing items", unit="item"):
        questions = item.get("question", [])
        name_fields = item.get("name", [])

        # 对每个 question 进行处理
        for question in questions:
            if prompt_type == "api" and api_info:
                # 从 name_fields 获取对应的 API 名称
                api_names = [name_field for name_field in name_fields if name_field in api_info]
                # 利用 Qwen 模型拆解意图并传递 API 信息
                parsed_intents, response = parse_intent_with_api_qwen(question, model, tokenizer, api_info, api_names)
            else:
                parsed_intents, response = parse_intent_with_standard_qwen(question, model, tokenizer)
            
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
def test_models(input_file, api_info=None):
    models = {
        "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        # 可以在这里添加更多模型，例如 Qwen-14B 或 Qwen-72B
        # "Qwen-2.5-14B": "Qwen/Qwen-2.5-14B",
        # "Qwen-2.5-72B": "Qwen/Qwen-2.5-72B"
    }
    
    for model_name, model_path in models.items():
        # 测试标准 prompt 方案
        output_file_standard = f'results/output_standard_{model_name}.json'
        print(f"Processing with {model_name} (Standard Prompt)...")
        process_intents(input_file, output_file_standard, model_path, prompt_type="standard")
        print(f"Results saved to {output_file_standard}")

        # 测试带有 API 信息的 prompt 方案
        if api_info:
            output_file_api = f'results/output_api_{model_name}.json'
            print(f"Processing with {model_name} (API Prompt)...")
            process_intents(input_file, output_file_api, model_path, api_info=api_info, prompt_type="api")
            print(f"Results saved to {output_file_api}")

# 调用测试函数
input_file = 'data/多意图数据(部分9.25).json'  # 输入文件路径
api_file = 'data/api_info.json'  # API 信息文件路径
api_info = read_api_info("data/new-samples-music.json", "data/new-samples-navigation.json", "new-samples-video.json", "new-samples-wechat.json")  # 从文件中读取 API 信息
test_models(input_file, api_info)
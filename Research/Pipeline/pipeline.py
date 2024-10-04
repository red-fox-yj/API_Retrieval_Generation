import difflib
import json
import os
import re
import numpy as np
import faiss
from tqdm import tqdm
from models import ModelHandlerV1
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_api_info(file_path):
    """
    从 JSON 文件中读取 API 名称和描述。
    :param file_path: JSON 文件路径
    :return: 包含 API 名称和描述的字典
    """
    api_info = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        api_data = json.load(f)
        # 遍历每个 API 并将其名称和描述添加到字典
        for api in api_data:
            api_name = api.get('name')
            api_description = api.get('description', 'No description available')
            if api_name:
                api_info[api_name] = api_description
    return api_info


def load_qwen_model(model_name):
    """
    加载 Qwen 模型和分词器
    :param model_name: 模型的名称或路径
    :return: 模型和分词器对象
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def read_qas(file):
    """
    从文件中读取 QA 数据
    :param file: JSON 文件路径
    :return: QA 数据列表
    """
    with open(file, 'r', encoding='utf-8') as f:
        qas = json.load(f)
    return qas


def create_faiss_index(embeddings):
    """
    创建 Faiss 索引
    :param embeddings: 向量化后的数据
    :return: Faiss 索引对象
    """
    print(f"Creating Faiss index with {embeddings.shape[0]} embeddings...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离创建索引
    index.add(embeddings)  # 添加向量
    print("Faiss index created.")
    return index


def search_similar_qa(embedding_test, index, test_qas, k=3, dedup=True):
    """
    使用 Faiss 索引检索与查询最相似的 QA
    :param embedding_test: 测试嵌入向量
    :param index: Faiss 索引
    :param test_qas: 测试集 QA 列表
    :param k: 返回的相似结果数量
    :param dedup: 是否去重
    :return: 相似的 QA 列表
    """
    unique_results = []
    seen_questions = set()
    offset = 0

    # 循环检索，直到获取到 k 个唯一的结果
    while len(unique_results) < k:
        distances, indices = index.search(embedding_test, k + offset)
        results = [test_qas[idx] for idx in indices[0]]

        # 去重处理
        for result in results:
            if dedup and result['Q'] not in seen_questions:
                seen_questions.add(result['Q'])
                unique_results.append(result)
            elif not dedup:
                unique_results.append(result)

            if len(unique_results) >= k:
                break

        offset += k

    return unique_results[:k]


def save_json_file(data, output_file):
    """
    将数据保存为 JSON 文件
    :param data: 要保存的数据
    :param output_file: 输出文件路径
    """
    dir_name = os.path.dirname(output_file)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def generate_instruction_and_prompt(test_qa):
    instruction = f"""
你是一个调用 API 解决问题的专家。你将会得到一些 API，以及其用法的描述。基于用户的问题，你需要进行有用的函数调用。你只需要给出单独的调用过程，形如 api_name(key = value)。
"""

    # 获取用户问题
    question = test_qa.get("Q")

    # 动态生成 API 列表
    apis = [api_des.get("name") for api_des in test_qa.get("few_shot", [])]
    apis_list = str(apis).replace("'", "")  # 格式化 API 列表
    
    # 初始化 prompt
    instruction += f"""
\n\n你可以使用以下 API {apis_list}，这些 API 的介绍如下：
"""

    # 添加每个 API 的描述
    for shot in test_qa.get("few_shot", []):
        api_name = shot.get("name")
        api_des = shot.get("des")
        instruction += f"\n名称: {api_name}\n描述: {api_des}"

    # 添加 few_shot 示例
    instruction += "\n\n示例:"
    for example in test_qa.get("few_shot", []):
        example_question = example.get("Q")
        example_answer = example.get("A")
        instruction += f"\n用户请求:{example_question}\n输出:{example_answer}\n"

    # 添加用户请求
    prompt = f"用户请求:{question}\n输出:"

    return instruction, prompt


def qwen_generate(instruction, prompt, model, tokenizer):
    """
    使用 Qwen 模型生成响应
    :param instruction: 系统指令
    :param prompt: 用户输入
    :param model: Qwen 模型
    :param tokenizer: 模型的分词器
    :return: 模型生成的响应文本
    """
    messages = [
        {"role": "system", "content": instruction},
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


def extract_function_info(func_str):
    """
    从函数字符串中提取函数名称和参数
    :param func_str: 函数调用字符串
    :return: 函数名和参数字典
    """
    # 匹配函数名和括号内的参数字符串
    func_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_\.]*)\((.*)\)')
    # 匹配参数键值对（参数值可以是带引号的字符串或无引号的内容）
    param_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(["\']?.*?["\']?)')

    match = func_pattern.match(func_str)
    if not match:
        raise ValueError("无法提取函数和参数")

    api_name, params_str = match.groups()

    # 解析参数字符串
    api_param = {}
    for param_match in param_pattern.finditer(params_str):
        param_key, param_value = param_match.groups()
        # 去除参数值两边的引号（如果有的话）
        param_value = param_value.strip('\'"')
        api_param[param_key] = param_value

    return api_name, api_param


def rewrite_query(query, model, tokenizer):
    """
    使用 LLM 改写用户的请求，将其生成多个不同的表达形式。
    :param query: 用户的原始请求
    :param model: 用于生成改写的模型
    :param tokenizer: 用于编码的分词器
    :return: 改写后的多种请求表达
    """
    instruction = """
    你是一个改写助手，请将以下请求改写成3种不同的表达方式。

    输出格式:
    改写1
    改写2
    改写3

    示例:
    原始请求: 还有多久能到达导航目的地

    输出:
    到达目的地还要多长时间？
    我们离目的地还有多长时间？
    预计什么时候可以到达？
    """

    # 拼接 prompt
    prompt = f"原始请求: {query}\n\n" \
             f"输出:"

    # 生成改写
    response = f"{query}\n{qwen_generate(instruction, prompt, model, tokenizer)}"
    
    return response


def parse_query(query, model, tokenizer):
    """
    解析用户请求，从自然语言中提取关键信息（如动作、目标、属性等）。
    :param test_qa: 包含用户问题的测试 QA
    :param model: 用于生成推理的模型
    :param tokenizer: 用于编码的分词器
    :return: 提取的信息 (动作, 目标, 相关属性)
    """
    instruction = """
    你是一个提取关键信息的智能助手，请从下面用户请求中提取关键信息。

    提取的关键信息包括：
    1. 动作（例如：播放歌曲、查询时间、查询路程、搜索、打开、关闭、跳转、推荐等）
    2. 目标（例如：歌曲、专辑、视频、车门、导航目的地、时间、路程等）
    3. 用户偏好（例如：最喜欢的，最近的，收藏等）
    4. 相关属性（例如：歌手、歌曲名、专辑名、时间等）

    输出格式:
    动作: <action>
    目标: <target>
    用户偏好: <preference>
    相关属性: <attribute_name> = <attribute_value>, <attribute_name> = <attribute_value>, ...

    示例1:
    用户请求: 
    播放王菲的《红豆》

    输出:
    动作: 播放歌曲
    目标: 歌曲
    用户偏好: 无
    相关属性: 歌曲名 = 红豆, 歌手 = 王菲

    示例2:
    用户请求: 
    打开我最喜欢的歌单<日本流行>

    输出:
    动作: 打开歌单
    目标: 歌单
    用户偏好: 最喜欢的
    相关属性: 歌单 = 日本流行
    """

    prompt = f"用户请求: {query}\n{rewrite_query(query, model, tokenizer)}\n\n" \
             f"输出:"

    response = qwen_generate(instruction, prompt, model, tokenizer)

    return response


def generate_api_info_prompt(few_shot):
    """
    动态生成 API 信息部分的提示
    :param api_info: 包含 API 名称和描述的字典
    :return: API 信息的字符串，用于插入到提示中
    """
    api_prompt = ""
    for shot in few_shot:
        api_prompt += f"- 名称: {shot['name']}\n  描述: {shot['api_des']}\n"
    return api_prompt


def select_most_similar_api(response, few_shot):
    """
    选择与 response 最相似的 API 名称。
    :param response: 模型生成的 API 名称
    :param few_shot: 包含 API 名称和描述的字典
    :return: 最相似的 API 名称
    """
    # 提取所有 API 的名称
    api_names = [shot['name'] for shot in few_shot]
    
    # 计算 response 与每个 API 名称的相似度
    similarities = [difflib.SequenceMatcher(None, response, api_name).ratio() for api_name in api_names]
    
    # 获取相似度最高的 API 名称
    most_similar_index = similarities.index(max(similarities))
    most_similar_api = api_names[most_similar_index]
    
    return most_similar_api

def select_api(parsed_query, few_shot, model, tokenizer):
    """
    选择合适的 API 函数。
    :param parsed_info: 从用户请求中提取出的关键信息 (动作, 目标, 属性)
    :param model: 用于生成推理的模型
    :param tokenizer: 用于编码的分词器
    :return: 选择的 API 函数
    """

    instruction = """
    你是一个根据关键信息选择 API 的智能助手，请根据下面的关键信息，选择合适的 API。

    输出格式:
    <api_name>

    示例:
    关键信息: 
    动作: 搜索歌曲
    目标: 歌曲
    相关属性: 歌曲名 = 红豆, 歌手 = 王菲

    你可以使用下面的 API:
    - 名称: music.search
      描述: 搜索歌曲，来首歌，我想听歌
    - 名称: video.search
      描述: 搜索视频，查找电影
    
    输出:
    music.search
    """

    # API 提示信息在 prompt 中填充
    prompt = f"关键信息:\n{parsed_query}\n\n" \
             f"你可以使用下面的 API:\n{generate_api_info_prompt(few_shot)}\n\n" \
             f"输出:"

    response = qwen_generate(instruction, prompt, model, tokenizer)

    # 确保 api_name 至少匹配到一个函数
    best_match_api = select_most_similar_api(response.strip(), few_shot)

    return best_match_api


def fill_api_parameters(api_name, parsed_query, few_shot, model, tokenizer):
    """
    基于选择的 API 函数和提取的属性，生成 API 调用。
    :param api_name: 选择的 API 名称
    :param parsed_info: 从用户请求中提取出的关键信息
    :param api_info: 包含 API 描述信息的字典
    :param model: 用于生成推理的模型
    :param tokenizer: 用于编码的分词器
    :return: API 调用字符串
    """

    instruction = """
    你是一个填充 API 参数的智能助手，请参考下面的 API 的调用示例，根据关键信息，填充 API 的参数，生成 API 调用信息，注意不要填充关键信息之外的参数。

    输出格式:
    api_name(key = value)
    
    示例:
    API 名称:
    music.search

    API 描述和参数信息:
    # 搜索歌曲，来首歌，我想听歌
    def music.search(
        song: string,          # 歌曲名
        musician: string,      # 歌手
        list: string,          # 歌单
        album: string,         # 专辑
        ranking: string,       # 排行榜
        instrument: string,    # 乐器
        style: string,         # 风格
        language: string,      # 语言
        times: string,         # 年代
        emotion: string,       # 心情
        scene: string,         # 场景
        gender: string,        # 性别
        free: boolean          # 是否免费
    ):

    API 调用示例:
    music.search(musician = 王菲, album = 天空)

    关键信息: 
    动作: 播放歌曲
    目标: 歌曲
    相关属性: 歌曲名 = 红豆, 歌手 = 王菲

    输出:
    music.search(song='红豆', musician='王菲')
    """
    sample = {}
    for shot in few_shot:
        if(shot['name'] == api_name):
            sample = shot
            break

    # 在 prompt 中填充需要的参数信息
    prompt = f"API 名称:\n{sample['name']}\n\n" \
             f"API 描述和参数信息:\n{sample['api_des']}\n\n" \
             f"API 调用示例:\n{sample['A']}\n\n" \
             f"关键信息:\n{parsed_query}\n\n" \
             f"输出:"

    response = qwen_generate(instruction, prompt, model, tokenizer)
    return response


def match_similarity(dict1, dict2):
    """
    计算两个字典的键和值的相似度。只有当键完全匹配时才计算值的相似度。
    :param dict1: 第一个字典
    :param dict2: 第二个字典
    :return: (键相似度, 值相似度)
    """
    
    # 计算键的相似度
    common_keys = set(dict1.keys()).intersection(set(dict2.keys()))  # 共同的键
    total_keys = set(dict1.keys()).union(set(dict2.keys()))  # 键的并集

    if len(total_keys) == 0:
        # 如果没有任何键，则返回相似度为 1
        return 1, 1

    key_similarity = len(common_keys) / len(total_keys)  # 键相似度，基于交集和并集

    # 如果键完全相同，计算值的相似度
    if key_similarity == 1:
        matched_values = 0
        for key in dict1:
            if dict1[key] == dict2[key]:
                matched_values += 1
        value_similarity = matched_values / len(dict1)  # 值相似度，按匹配的值比例计算
    else:
        value_similarity = 0  # 如果键不完全匹配，值相似度直接为 0

    return key_similarity, value_similarity


def infer_template1(test_qa, model, tokenizer):
    """
    基于用户的测试数据生成响应。

    此方法执行以下步骤：
    1. 生成用于推理的指令和提示。
    2. 调用 Qwen 模型生成响应。
    
    :param test_qa: dict，包含用户问题和API候选项的测试数据。
    :param model: LLM 模型，用于生成响应。
    :param tokenizer: 分词器，负责对输入和输出进行编码与解码。
    :return: tuple，返回两次相同的响应结果（用于后续处理的扩展）。
    """
    
    instruction, prompt = generate_instruction_and_prompt(test_qa)  # 生成指令和提示
    response = qwen_generate(instruction, prompt, model, tokenizer)  # 调用模型生成响应
    return response, response  # 返回相同的响应


def infer_template2(test_qa, model, tokenizer):
    """
    处理用户请求，包含以下步骤：
    1. 解析用户请求的关键信息
    2. 选择合适的 API
    3. 根据解析结果填写 API 参数并生成最终调用

    :param test_qa: 包含用户请求的测试数据
    :param model: 用于生成推理的模型
    :param tokenizer: 用于编码的分词器
    :return: 最终的 API 调用和整个响应过程
    """

    # Step 1: 解析用户请求
    parsed_query = parse_query(test_qa['Q'], model, tokenizer)

    # Step 2: 选择 API 函数
    selected_api = select_api(parsed_query, test_qa['few_shot'], model, tokenizer)

    # Step 3: 填写 API 参数并生成调用
    final_api_call = fill_api_parameters(selected_api, parsed_query, test_qa['few_shot'], model, tokenizer)

    response = f"解析后的请求: {parsed_query};\n选择的 API: {selected_api};\n最终 API 调用: {final_api_call}"

    return final_api_call, response


def infer_template3(test_qa, model, tokenizer):
    """
    处理用户请求，包含以下步骤：
    1. 改写用户请求
    2. 解析用户请求的关键信息
    3. 选择合适的 API
    4. 根据解析结果填写 API 参数并生成最终调用

    :param test_qa: 包含用户请求的测试数据
    :param model: 用于生成推理的模型
    :param tokenizer: 用于编码的分词器
    :return: 最终的 API 调用和整个响应过程
    """
    
    # Step 1: 改写用户请求
    rewrited_query = rewrite_query(test_qa['Q'], model, tokenizer)

    # Step 2: 解析用户请求
    parsed_query = []
    for query in rewrited_query:
        parsed_query.append(parse_query(query, model, tokenizer))

    # Step 3: 选择 API 函数
    selected_api = select_api(parsed_query, test_qa['few_shot'], model, tokenizer)

    # Step 4: 填写 API 参数并生成调用
    final_api_call = fill_api_parameters(selected_api, parsed_query, test_qa['few_shot'], model, tokenizer)

    response = f"改写后的请求: {rewrited_query};\n解析后的请求: {parsed_query};\n选择的 API: {selected_api};\n最终 API 调用: {final_api_call}"

    return final_api_call, response


def single_pipeline(infer_template, embedding_model_name, generate_model_name, api_info_path, single_qas_path, test_size=0.5, k=3, dedup=False):
    """
    执行单管道流程，加载模型并进行 QA 匹配和推理
    :param model_name: 使用的模型名称
    :param api_info_path: API 信息文件路径
    :param single_qas_path: QA 数据文件路径
    :param test_size: 测试集比例
    :param k: Faiss 检索返回的相似项个数
    :param dedup: 是否去重
    """
    # 加载模型
    model_handler = ModelHandlerV1(embedding_model_name)

    # 读取 QA 数据和 API 信息
    qas = read_qas(single_qas_path)
    api_info = read_api_info(api_info_path)

    # 数据集拆分（间隔一个进行拆分）
    train_qas = qas[::2]  # 间隔一个元素放入训练集
    test_qas = qas[1::2]  # 间隔一个元素放入测试集

    # 提取问题并向量化
    train_qs = [qa['Q'] for qa in train_qas]
    test_qs = [qa['Q'] for qa in test_qas]
    train_q_embeddings = model_handler.get_batch_embeddings(train_qs)
    test_q_embeddings = model_handler.get_batch_embeddings(test_qs)

    # 创建 Faiss 索引
    index = create_faiss_index(train_q_embeddings)

    # 遍历测试集，检索 top_k 结果
    
    for test_qa, test_q_embedding in zip(test_qas, test_q_embeddings):
        test_q_embedding_np = np.array(test_q_embedding).reshape(1, -1)
        top_k_results = search_similar_qa(test_q_embedding_np, index, train_qas, k, dedup)

        # 更新 API 描述和 few_shot
        test_qa['few_shot'] = []
        for result in top_k_results:
            result['api_des'] = api_info[result['name']]
            test_qa['few_shot'].append(result)

    # 保存结果
    save_json_file(test_qas, f"results/retrieval_{embedding_model_name.replace('/', '_')}_{generate_model_name.replace('/', '_')}_test_size_{test_size}_k_{k}_dedup_{dedup}.json")

    # 加载模型
    model, tokenizer = load_qwen_model(generate_model_name)

    success_retrieval_num = 0
    success_format_num = 0
    success_api_name_num = 0
    success_api_param_key_num = 0
    success_api_param_value_num = 0
    error_log = []

    # 遍历测试集进行推理
    for test_qa in tqdm(test_qas, desc="Generating response"):
        # retrieval
        if any(test_qa['name'] == shot['name'] for shot in test_qa['few_shot']):
            success_retrieval_num += 1
        else:
            test_qa['error_type'] = 'retrieval_error'
            error_log.append(test_qa)
            continue

        final_api_call = ''
        response = ''

        
        if infer_template == 'infer_template3': # 改写 query + cot 推理
            final_api_call, response = infer_template3(test_qa, model, tokenizer)
        elif infer_template == 'infer_template2': # cot 推理
            final_api_call, response = infer_template2(test_qa, model, tokenizer)
        else: # 一次推理
            final_api_call, response = infer_template1(test_qa, model, tokenizer)

        # format
        try:
            api_name, api_param = extract_function_info(final_api_call)
            # 确保不会因为 api_name 拼写错误导致的指标下降
            api_name = select_most_similar_api(api_name.strip(), test_qa['few_shot'])
            success_format_num += 1
        except Exception:
            test_qa['error_type'] = 'format_error'
            test_qa['response'] = response
            error_log.append(test_qa)
            continue

        # api_name
        api_name_a, api_param_a = extract_function_info(test_qa['A'])
        if api_name == api_name_a:
            success_api_name_num += 1
        else:
            test_qa['error_type'] = 'api_name_error'
            test_qa['response'] = response
            error_log.append(test_qa)
            continue

        # api_param
        success_api_param_key_similarity, success_api_param_value_similarity = match_similarity(api_param_a, api_param)
        success_api_param_key_num += success_api_param_key_similarity
        success_api_param_value_num += success_api_param_value_similarity

        if success_api_param_key_similarity != 1:
            test_qa['error_type'] = 'api_param_key_error'
            test_qa['response'] = response
            error_log.append(test_qa)
            continue

        if success_api_param_value_similarity != 1:
            test_qa['error_type'] = 'api_param_value_error'
            test_qa['response'] = response
            error_log.append(test_qa)
            continue
        
        # 没有错误也保存结果
        test_qa['response'] = response
        error_log.append(test_qa)

    # 保存失败日志
    save_json_file(error_log, f"results/{infer_template}_error_{embedding_model_name.replace('/', '_')}_{generate_model_name.replace('/', '_')}_test_size_{test_size}_k_{k}_dedup_{dedup}.json")

    # 计算成功率
    success_retrieval_ratio = success_retrieval_num / len(test_qas)
    success_format_ratio = success_format_num / success_retrieval_num
    success_api_name_ratio = success_api_name_num / success_format_num
    success_api_param_key_ratio = success_api_param_key_num / success_api_name_num
    success_api_param_value_ratio = success_api_param_value_num / success_api_param_key_num

    # 保存评估结果
    save_json_file({
        "success_retrieval_ratio": success_retrieval_ratio,
        "success_format_ratio": success_format_ratio,
        "success_api_name_ratio": success_api_name_ratio,
        "success_api_param_key_ratio": success_api_param_key_ratio,
        "success_api_param_value_ratio": success_api_param_value_ratio
    }, f"results/{infer_template}_metric_{embedding_model_name.replace('/', '_')}_{generate_model_name.replace('/', '_')}_test_size_{test_size}_k_{k}_dedup_{dedup}.json")

    print(f"{infer_template} end pipeline.")


if __name__ == "__main__":
    single_pipeline('infer_template1', 'nvidia/NV-Embed-v2','Qwen/Qwen2.5-7B-Instruct', 'data/api_info.json', 'data/single_qas.json', test_size=0.5, k=3, dedup=False)
    single_pipeline('infer_template2', 'nvidia/NV-Embed-v2','Qwen/Qwen2.5-7B-Instruct', 'data/api_info.json', 'data/single_qas.json', test_size=0.5, k=3, dedup=False)
    single_pipeline('infer_template3', 'nvidia/NV-Embed-v2','Qwen/Qwen2.5-7B-Instruct', 'data/api_info.json', 'data/single_qas.json', test_size=0.5, k=3, dedup=False)

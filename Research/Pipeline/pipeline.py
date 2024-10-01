import json
import os
import re
import numpy as np
import faiss  # 请确保 Faiss 已安装
from Research.pipeline.models import ModelHandlerV1
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
    """
    你是一个调用函数工具解决问题的专家。你将会得到一些函数工具，以及其用法的描述。基于用户的问题，你需要进行有用的函数调用。你只需要给出单独的调用过程，形如 {"name":api_name, "parameters":{"key":"value"}}。

    你可以使用以下工具[music.search, music.recent.open, music.favorite.open, video.search]，这些工具的介绍如下：

    名称: music.search
    描述: # 搜索歌曲，来首歌，我想听歌
    def music.search(song: string, musician: string, list: string, album: string, ranking: string, instrument: string, style: string, language: string, times: string, emotion: string, scene: string, gender: string, free: boolean):
    # song: 歌曲名
    # musician: 歌手
    # list: 歌单
    # album: 专辑
    # ranking: 排行榜
    # instrument: 乐器
    # style: 风格
    # language: 语言
    # times: 年代
    # emotion: 心情
    # scene: 场景
    # gender: 性别
    # free: 是否免费

    名称: video.search
    描述: # 搜索视频
    def video.search(video: string, starring: string, domain: string, free: boolean, up: string):
    # video: 视频名
    # starring: 主演
    # domain: 类型
    # free: 是否免费
    # up: up主

    示例:

    我想听周杰伦的《七里香》
    {"name": "music.search", "parameters": music.search(song = 七里香, musician = 周杰伦)}

    有没有斯嘉丽·约翰逊主演的《黑寡妇》可以看？
    {"name": "video.search", "parameters": video.search(video = 黑寡妇, starring = 斯嘉丽·约翰逊)}

    用户请求:
    播放王菲的《红豆》
    """
    instruction = """
你是一个调用函数工具解决问题的专家。你将会得到一些函数工具，以及其用法的描述。基于用户的问题，你需要进行有用的函数调用。你只需要给出单独的调用过程，形如 {"name":api_name, "parameters":{"key":"value"}}。
"""
    # 获取用户问题
    question = test_qa.get("Q")

    # 动态生成 API 列表
    apis = [api_description.get("name") for api_description in test_qa.get("api_des", [])]
    apis_list = str(apis).replace("'", "")  # 格式化 API 列表
    
    # 初始化 prompt
    prompt = f"""
你可以使用以下工具{apis_list}，这些工具的介绍如下：
"""

    # 添加每个 API 的描述
    for api_description in test_qa.get("api_des", []):
        api_name = api_description.get("name")
        api_description_text = api_description.get("description")
        prompt += f"\n名称: {api_name}\n描述: {api_description_text}"

    # 添加 few_shot 示例
    prompt += "\n\n示例:\n"
    for example in test_qa.get("few_shot", []):
        example_name = example.get("name")
        example_question = example.get("Q")
        example_answer = example.get("A")
        prompt += f"\n{example_question}\n{{\"name\": \"{example_name}\", \"parameters\": {example_answer}}}"

    # 添加用户请求
    prompt += f"\n\n用户请求:\n{question}\n"
    
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
    # 生成输入文本
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # 生成响应
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    # 解码生成的响应
    generated_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_response


def extract_function_info(func_str):
    """
    从函数字符串中提取函数名称和参数
    :param func_str: 函数调用字符串
    :return: 函数名和参数字典
    """
    func_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_\.]*)\((.*)\)')
    param_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(".*?"|\'.*?\'|[^,)]+)')

    match = func_pattern.match(func_str)
    if not match:
        raise ValueError("无法提取函数和参数")

    func_name, params_str = match.groups()

    # 解析参数字符串
    params = {}
    for param_match in param_pattern.finditer(params_str):
        param_key, param_value = param_match.groups()
        if param_value.startswith(("'", '"')) and param_value.endswith(("'", '"')):
            param_value = param_value[1:-1]
        params[param_key] = param_value

    return func_name, params


def match_ratio(dict1, dict2):
    """
    计算两个字典的匹配比例
    :param dict1: 第一个字典
    :param dict2: 第二个字典
    :return: 匹配比例
    """
    match_count = 0
    for key in dict1:
        if key in dict2 and dict1[key] == dict2[key]:
            match_count += 1

    return match_count / len(dict1) if len(dict1) > 0 else 0


def single_pipeline(model_name, api_info_path, single_qas_path, test_size=0.5, k=3, dedup=False):
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
    model_handler = ModelHandlerV1(model_name)

    # 读取 QA 数据和 API 信息
    qas = read_qas(single_qas_path)
    api_info = read_api_info(api_info_path)

    # 数据集拆分
    split_idx = int(len(qas) * test_size)
    train_qas, test_qas = qas[:split_idx], qas[split_idx:]

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
        test_qa['api_des'] = []
        test_qa['few_shot'] = []
        for result in top_k_results:
            test_qa['api_des'].append(api_info[result['name']])
            test_qa['few_shot'].append(result)

    # 保存结果
    save_json_file(test_qas, f"results/multi_pipeline_{model_name.replace('/', '_')}_test_size_{test_size}_k_{k}_dedup_{dedup}.json")

    # 加载模型
    model, tokenizer = load_qwen_model('Qwen/Qwen2.5-7B-Instruct')

    success_format_num = 0
    success_api_name_num = 0
    success_api_param_num = 0
    error_log = []

    # 遍历测试集进行推理
    for test_qa in test_qas:
        instruction, prompt = generate_instruction_and_prompt(test_qa)
        response = qwen_generate(instruction, prompt, model, tokenizer)

        # 评估结果
        try:
            api_name, api_param = extract_function_info(response)
            success_format_num += 1
        except Exception:
            test_qa['error_type'] = 'format_error'
            test_qa['response'] = response
            error_log.append(test_qa)
            continue

        # 比较 API 名称和参数
        api_name_a, api_param_a = extract_function_info(test_qa['A'])
        if api_name != api_name_a:
            test_qa['error_type'] = 'api_name_error'
            test_qa['response'] = response
            error_log.append(test_qa)
            continue

        api_param_match_ratio = match_ratio(api_param_a, api_param)
        if api_param_match_ratio != 1:
            test_qa['error_type'] = 'api_param_error'
            test_qa['response'] = response
            error_log.append(test_qa)
        
        success_api_param_num += api_param_match_ratio

    # 计算成功率
    success_format_ratio = success_format_num / len(test_qas)
    success_api_name_ratio = success_api_name_num / len(test_qas)
    success_api_param_ratio = success_api_param_num / len(test_qas)

    # 保存评估结果
    save_json_file({
        "success_format_ratio": success_format_ratio,
        "success_api_name_ratio": success_api_name_ratio,
        "success_api_param_ratio": success_api_param_ratio
    }, f"results/pipeline_{model_name.replace('/', '_')}_test_size_{test_size}_k_{k}_dedup_{dedup}.json")


if __name__ == "__main__":
    # 执行单管道流程
    single_pipeline('Qwen/Qwen2.5-7B-Instruct', 'data/api_info.json', 'data/single_qas.json', test_size=0.5, k=3, dedup=False)

import json
import os


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_file(data, output_file):
    """
    将数据保存为指定的 JSON 文件。如果输出目录不存在，则创建它。
    :param data: 要保存的数据
    :param output_file: 输出的文件路径
    """
    dir_name = os.path.dirname(output_file)
    
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def extract_api_description(data, api_descriptions):
    """
    提取 API 描述并将其合并到已有的描述列表中
    :param data: JSON 数据
    :param api_descriptions: 已有的 API 描述列表，将新描述添加到这个列表中
    """
    for item in data:
        if 'name' in item and 'description' in item:
            api_descriptions.append({
                'name': item['name'],
                'description': item['description']
            })


def extract_qa(data, qa_data):
    """
    提取 qa 部分并将其合并到已有的 QA 列表中，附加对应的 API name 信息
    :param data: JSON 数据
    :param qa_data: 已有的 QA 列表，将新 QA 项目添加到这个列表中
    """
    for item in data:
        if 'qa' in item and 'name' in item:
            for qa_item in item['qa']:
                qa_data.append({
                    'name': item['name'],  # 附加对应的 name 信息
                    'Q': qa_item['Q'],
                    'A': qa_item['A']
                })


def process_single_file(file_path, api_descriptions, qa_data):
    """
    处理单个 JSON 文件，将提取出的 API 描述和 QA 部分分别合并到已有的列表中
    :param file_path: 输入的 JSON 文件路径
    :param api_descriptions: 已有的 API 描述列表
    :param qa_data: 已有的 QA 数据列表
    """
    data = read_json_file(file_path)

    # 提取 API 描述
    extract_api_description(data, api_descriptions)

    # 提取 QA 数据
    extract_qa(data, qa_data)


def extract_single_qas_and_api_info(input_files, single_qas_filename, api_info_filename):
    api_info = []  # 用于存储所有文件的 API 描述
    single_qas = []  # 用于存储所有文件的 QA 数据

    for file_path in input_files:
        process_single_file(file_path, api_info, single_qas)

    # 将合并的 API 描述保存到单个文件
    save_json_file(api_info, api_info_filename)
    print(f"所有 API 描述已保存到: {api_info_filename}")

    # 将合并的 QA 数据保存到单个文件
    save_json_file(single_qas, single_qas_filename)
    print(f"所有 QA 数据已保存到: {single_qas_filename}")


def extract_muti_qas(input_file, output_file):
    # 读取 JSON 文件
    data = read_json_file(input_file)

    # 提取 'name' 和 'question' 字段，按name分组
    muti_qas = []

    # 遍历每个数据项
    for item in data:
        for answer_item in item['answer']:
            muti_qas.append({
                'name': answer_item['name'],
                'question': answer_item['question']
            })

    # 保存训练集和测试集
    save_json_file(muti_qas, output_file)
    print(f"muti_qas 保存到: {output_file}")


# 提取所有单意图数据和API描述数据分别单独保存为文件
extract_single_qas_and_api_info(
    ["data/new-samples-music.json", "data/new-samples-navigation.json", "data/new-samples-video.json", "data/new-samples-wechat.json"], 
    'data/single_qas.json',  # 保存合并后的 QA 文件名
    'data/api_info.json'  # 保存合并后的 API 描述文件名
)


# 调用主函数进行提取、划分和保存
extract_muti_qas('data/多意图数据(部分9.25).json', 'data/muti_qas.json')

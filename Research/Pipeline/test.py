print("Begin Runing")
from datetime import date
import argparse
import json
from tqdm import tqdm
import random
import re
print("Importing VLLM")
from vllm import LLM, SamplingParams
import faiss
import os
from models import ModelHandlerV1, ModelHandlerV2, ModelHandlerV3
import numpy as np

random.seed(1024)
print("Import Finished")
MODEL_CLASS = ModelHandlerV1
MODEL_NAME = 'nvidia/NV-Embed-v2'
TEST_PROP = 0.5

TEMP_DATASET = "uniformed_data_code.jsonl" 
BATCH_SIZE = 32
DEBUG = 0
FEW_SHOT = 3 # 示例数
NEAREST_K = 4 # 检索工具数
PYTHON_FORMAT = True # 是不是调用变成严格的python调用格式（加引号）

DATASET_LIST = [
    "data_code/new-samples-music.json",
    "data_code/new-samples-navigation.json",
    "data_code/new-samples-video.json",
    "data_code/new-samples-wechat.json"
]
"""
srun --gres=gpu:1 --cpus-per-task=12 -p a800 --pty python code_format_tester.py \
--model_path=Qwen/Qwen2.5-7B-Instruct  --msg=test_qwen25_7B_my_template

srun --gres=gpu:1 --cpus-per-task=12 -p fnlp-x090 --pty python code_format_tester.py \
--model_path=Qwen/Qwen2-7B-Instruct  --msg=test_qwen2_7B_my_template

srun --gres=gpu:1 --cpus-per-task=12 -p fnlp-x090 --pty python code_format_tester.py \
--model_path=/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B-Instruct  --msg=test_llama_31_8B_my_template_31
"""

print("Library imported")


def extract_function_info(func_str):
    # 正则表达式匹配函数名和参数
    func_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_\.]*)\((.*)\)')
    param_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(".*?"|\'.*?\'|[^,)]+)')

    # 查找函数名和参数字符串
    match = func_pattern.match(func_str)
    if not match:
        raise ValueError("Can't extract function and parameters")

    func_name, params_str = match.groups()

    # 解析参数字符串
    params = {}
    for param_match in param_pattern.finditer(params_str):
        param_key, param_value = param_match.groups()
        # 移除参数值两边的引号（如果有的话）
        if param_value.startswith(("'", '"')) and param_value.endswith(("'", '"')):
            param_value = param_value[1:-1]
        params[param_key] = param_value

    return {"name": func_name, "parameters": params}

def dict_to_function_call(func_dict):
    # 获取函数名
    func_name = func_dict['name']
    
    # 获取参数字典
    params = func_dict['parameters']
    
    # 准备参数字符串列表
    param_str_list = []
    for key, value in params.items():
        # 如果值是布尔值，则直接添加
        if value == "True" or value == "False" or re.match(r'^-?\d+(\.\d+)?$', value):
            param_str_list.append(f"{key}={value}")
        # 否则，将值视为字符串，并添加引号
        else:
            param_str_list.append(f"{key}=\"{value}\"")
    
    # 将参数字符串用逗号分隔并拼接到函数名后面
    function_call_str = f"{func_name}({', '.join(param_str_list)})"
    
    return function_call_str

def call_format(func_string):
    return dict_to_function_call(extract_function_info(func_string))

# 导入数据，直接进行分割，并且把train集合上的直接当作few_shot保留一下。以及各个name的name_to_doc也要保留一下
def load_data_and_split(test_prop, few_shot):
    name_desc = {}
    qa_train = []
    qa_test = []
    for file_path in DATASET_LIST:
        print(f"Loading {file_path}")
        with open(file_path, encoding="utf-8") as f:
            tools = json.load(f)
        for tool in tools:
            tool_qa_pairs = tool.get("qa", [])
            if PYTHON_FORMAT:
                for qa in tool_qa_pairs:
                    qa["A"] = call_format(qa["A"])
            random.shuffle(tool_qa_pairs)
            length = len(tool_qa_pairs)
            tool_qa_test = tool_qa_pairs[:min(round(length*test_prop), length-FEW_SHOT)]
            tool_qa_train = tool_qa_pairs[min(round(length*test_prop), length-FEW_SHOT):]
            name_desc[tool["name"]] = f'{tool["name"]}\n' + f'{tool["description"]}\n' + ("示例:" if FEW_SHOT!=0 else "") \
                + "\n".join(f'{qa["Q"]}\n{qa["A"]}' for qa in tool_qa_train[:few_shot])
            qa_test.extend(tool_qa_test)
            qa_train.extend(tool_qa_train)
    return qa_test, qa_train, name_desc

def get_questions_embedding(model_handler, questions):
    embeddings = model_handler.get_batch_embeddings(questions)
    return embeddings

def create_faiss_index(embeddings):
    """创建 Faiss 索引"""
    print(f"Creating Faiss index with {embeddings.shape[0]} embeddings...")
    dimension = embeddings.shape[1]  # 嵌入向量的维度
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("Faiss index created.")
    return index

def search_similar_tool(query_embedding, index, qa_pairs, k=3, dedup=True):
    """检索与输入查询最相似的 QA 对，支持去重"""
    unique_results = []
    seen_questions = set()
    offset = 0
    
    # 循环，直到获得 k 个唯一的结果
    while len(unique_results) < k:
        distances, indices = index.search(query_embedding, k + offset)
        results = [qa_pairs[idx]["A"].split("(")[0] for idx in indices[0]]
        
        # 去重：只添加还未见过的 `Q`
        for result in results:
            if dedup and result not in seen_questions:
                seen_questions.add(result)
                unique_results.append(result)
            elif not dedup:
                unique_results.append(result)
                
            if len(unique_results) >= k:
                break

        # 如果未找到足够的唯一结果，增加偏移量，继续检索更多
        offset += k

    return unique_results[:k]

def few_shot(qa_list):
    return "\n示例:\n"+"\n".join([f"Q: {qa['Q']}\nA: {qa['A']}" for qa in qa_list])+"\n"

def main(model, test_prop, few_shot, nearest_k):
    # 分割
    qa_test, qa_train, name_desc = load_data_and_split(test_prop=test_prop, few_shot=few_shot)
    print("Data loaded and splited")
    
    # 向量化
    test_question_embedding = get_questions_embedding(model, [qa["Q"] for qa in qa_test])
    qa_test = [{**elm, "embedding":embedding} for elm, embedding in zip(qa_test, test_question_embedding)]
    train_question_embedding = get_questions_embedding(model, [qa["Q"] for qa in qa_train])
    qa_train = [{**elm, "embedding":embedding} for elm, embedding in zip(qa_train, train_question_embedding)]
    print("Embedding got")
    
    # 向量数据库
    index = create_faiss_index(np.vstack([elm["embedding"] for elm in qa_train]))
    print("Index built")
    
    # 检索
    qa_pair_train = [{key: value for key, value in elm.items() if key != "embedding"} for elm in qa_train]
    qa_test = [{**qa, "tool_names":search_similar_tool(np.array(qa["embedding"]).reshape(1, -1), index, qa_pair_train, k=nearest_k)} for qa in qa_test]
    qa_test = [{**qa, "tool_docs":[name_desc[name] for name in qa["tool_names"]]} for qa in qa_test]
    data = []
    for elm in qa_test:
        data.append({
            "query": elm["Q"],
            "answer": elm["A"],
            "tools_doc": elm["tool_docs"],
            "tools_list": elm["tool_names"]
        })
    return data


class Qwen2ChatFormat:
    SYSTEM_PROMPT = (
    "你是一个调用函数工具解决问题的专家.。"
    "你将会得到一些函数工具，以及其用法的描述。"
    "基于用户的问题，你需要进行有用的函数调用。"
    "你只需要给出单独的调用过程，形如 tool_name(key1=value1, key2=value2) "
     )
    
    @staticmethod
    def get_header_prompt(message) -> str:
        return "<|im_start|>" + message["role"] + "\n"
    
    @staticmethod
    def post_process(output):
        return output

    @staticmethod
    def get_message_prompt(message) -> str:
        prompt = Qwen2ChatFormat.get_header_prompt(message)
        prompt += message["content"].strip() + "<|im_end|>"
        return prompt

    @staticmethod
    def get_dialog_prompt(dialog) -> str:
        prompt = ""
        for message in dialog:
            prompt += Qwen2ChatFormat.get_message_prompt(message)
        # Add the start of an assistant message for the model to complete.
        prompt += Qwen2ChatFormat.get_header_prompt({"role": "assistant", "content": ""})
        return prompt
    
    @staticmethod
    def dialog_template(query, tools_names, tools_doc):
        tool_description = f"你可以使用以下工具{tools_names},这些工具的介绍如下\n\n" + "\n\n".join(tools_doc)
        return [
            {
                "role": "system",
                "content": Qwen2ChatFormat.SYSTEM_PROMPT + '\n\n' + tool_description
            },
            {
                "role": "user",
                "content": query
            }
        ]

class Llama31ChatFormat:

    SYSTEM_PROMPT = (
        f"\n\nCutting Knowledge Date: December 2023\n"
        f"Today Date: {date.today().strftime('%d %b %Y')}\n\n"
        "When you receive a tool call response, use the output to format an answer to the original user question.\n"
        "You are a helpful assistant with tool calling capabilities."
    )

    USER_PROMPT_1 = (
    "Given the following functions, please respond with function call," 
    "with its proper arguments that best answers the given prompt."
    "You should only use arguments supported by user query."
    "Respond in the format tool_name(key1=value1, ...). Do not use variables."
    )
    
    USER_PROMPT_2 = (
        "The tools you can use are as following:"
    )


    @staticmethod
    def get_header_prompt(message) -> str:
        return "<|start_header_id|>" + message["role"] + "<|end_header_id|>\n\n"

    @staticmethod
    def get_message_prompt(message) -> str:
        prompt = Llama31ChatFormat.get_header_prompt(message)
        prompt += message["content"].strip() + "<|eot_id|>"
        return prompt

    @staticmethod
    def get_dialog_prompt(dialog) -> str:
        prompt = "<|begin_of_text|>"
        for message in dialog:
            prompt += Llama31ChatFormat.get_message_prompt(message)
        # Add the start of an assistant message for the model to complete.
        prompt += Llama31ChatFormat.get_header_prompt({"role": "assistant", "content": ""})
        return prompt

    @staticmethod
    def dialog_template(query, tools_names, tools_doc):
        return [
            {
                "role": "system",
                "content": Llama31ChatFormat.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": Llama31ChatFormat.USER_PROMPT_1  +"\n" +Llama31ChatFormat.USER_PROMPT_2 + str(tools_names)\
                    +"\n\n".join(tools_doc) + "\n\nQuestion: " + query
            }
        ]



def test_with(data_path, formatter, llm, sampling_params):
    result = []
    query_list = []
    tools_list_list = []
    try:
        debug = DEBUG
    except:
        debug = 0

    input_text_to_test, answer_to_text = [], []

    with open(data_path, 'r') as f:
        lines = f.readlines()
    # random.shuffle(lines)
    if debug > 0:
        random.shuffle(lines)
        lines = lines[: debug]
    for line in lines:
        data = json.loads(line)
        dialog = formatter.dialog_template(query=data["query"], tools_doc=data["tools_doc"], tools_names=data["tools_list"])
        input_text_to_test.append(formatter.get_dialog_prompt(dialog))
        answer_to_text.append(data["answer"])
        query_list.append(data["query"])
        tools_list_list.append(data["tools_list"])
    tool_num = len(data["tools_list"])
    few_shot = data["tools_doc"][0].count("(")-1 if len(data["tools_doc"])>0 else 0

    json_format = 0
    api_name = 0
    parameters_include = 0
    parameters_exact_match = 0
    tool_available = 0
    data_cnt = len(input_text_to_test)
    output_list = []
    batch_input = []

    if debug > 0:
        print(input_text_to_test[0])

    for i in tqdm(range(data_cnt)):
        batch_input.append(input_text_to_test[i])
        if len(batch_input) == BATCH_SIZE:
            output_list += [generated.outputs[0].text for generated in llm.generate(batch_input, sampling_params)]

            batch_input = []
    output_list += [generated.outputs[0].text for generated in llm.generate(batch_input, sampling_params)]
    result_list = []
    for input_text, output, answer, query_, tools_list in tqdm(
        zip(input_text_to_test, output_list, answer_to_text, query_list, tools_list_list), total=len(answer_to_text)
    ):
        result.append({
            "input_text": input_text,
            "query": query_,
            "target": answer,
            "output": output,
            
        })
        if debug > 0:
            print("*" * 40)
            # print(input_text)
            print(query_)
            print(answer)
            print(output)
            print("*" * 40)
        try:
            json_output = extract_function_info(output)
            
            answer = extract_function_info(answer)
            
            json_format += 1
            if debug > 0:
                print(answer)
                print(json_output)
                print("*" * 40)
        except Exception as e:
            if debug > 0:
                print(e)
            result_list.append([0, 0, 0, 0])
            continue
        try:
            if answer["name"] in tools_list:
                tool_available += 1
            if json_output["name"] == answer["name"]:
                api_name += 1
                
                if set(answer["parameters"].keys()) != set(json_output["parameters"].keys()):
                    para_unmatch = True
                else:
                    para_unmatch = False

                tmp_cnt, tmp_matched, tmp_smatch = 0, 0, 0
                for key, value in answer["parameters"].items():
                    tmp_cnt += 1
                    output_value = json_output["parameters"].get(key, "")
                    if output_value == value:
                        tmp_matched += 1
                    if str(output_value).strip() == str(value).strip():
                        tmp_smatch += 1
                if tmp_matched == tmp_cnt:
                    parameters_include += 1
                if tmp_smatch == tmp_cnt and not para_unmatch:
                    parameters_exact_match += 1
                result_list.append([1, 1, int(tmp_matched == tmp_cnt), int(not para_unmatch and tmp_matched == tmp_cnt)])
            else:
                result_list.append([1, 0, 0, 0])
        except Exception as e:
            result_list.append([1, 0, 0, 0])
        if debug > 0:
            print(result_list[-1])
        debug -= 1
        
    Retrived = round(tool_available / data_cnt * 100, 2)
    Parsed = round(json_format / data_cnt * 100, 2)
    APIAcc = round(api_name / data_cnt * 100, 2)
    ExactMatch = round(parameters_include / data_cnt * 100, 2)
    SEM = round(parameters_exact_match / data_cnt * 100, 2)

    print("含正确工具：{}({}%)".format(tool_available, Retrived))
    print("可被解析：{} ({}%)".format(json_format, Parsed))
    print("工具正确：{} ({}%)".format(api_name, APIAcc))
    print("参数子集：{} ({}%)".format(parameters_include, ExactMatch))
    print("参数全体：{} ({}%)".format(parameters_exact_match, SEM))
    print("示例数：{}".format(FEW_SHOT))
    print("工具数：{}".format(NEAREST_K))
    with open(f"output_record_{data_path}", "w", encoding="utf-8") as f:
        for elm in result:
            f.write(json.dumps(elm, ensure_ascii=False)+"\n")
    return Parsed, APIAcc, ExactMatch, SEM, result_list

def test_all(model_path, model_size, dataset, steps=0, tp=1, msg="", max_length=1024, demo_num=-1, api_num=-1):
    if "llama" in msg:
        formatter = Llama31ChatFormat
    else:
        formatter = Qwen2ChatFormat
    res_dict = {}
    data_path = TEMP_DATASET
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=max_length
    )
    sampling_params = SamplingParams(
        max_tokens=max_length, n=1, stop=["<|eot_id|>"], skip_special_tokens=False
    )  # temperature=0.95, top_p=0.95,
    
    report_dict = {
        "说明": msg,
        "模型参数": model_size,
        "steps": steps,
    }

    if demo_num != -1:
        report_dict["demo_num"] = demo_num
    if api_num != -1:
        report_dict["api_num"] = api_num
    if "_demo" in model_path:
        report_dict["trained_demo"] = int(model_path.split("_demo")[0].split("_")[-1])
    if "_retrieved" in model_path:
        report_dict["trained_api"] = int(model_path.split("_retrieved")[0].split("_")[-1])
        
    Parsed, APIAcc, ExactMatch, SEM, res = test_with(data_path, formatter, llm, sampling_params)
    report_dict = {
        **report_dict,
        "Parsed": Parsed,
        "Acc": APIAcc,
        "EM": ExactMatch,
        "SEM": SEM,
    }
    res_dict["res"] = res

    if api_num == -1:
        api_num = 4
    if demo_num == -1:
        demo_num = 0
    with open(f'logs/{msg}_api_{api_num}_demo_{demo_num}.json', "w") as file_out:
        json.dump(
            {
                **report_dict,
                **res_dict,
            }, file_out,
            ensure_ascii=False
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_size", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--demo_num", type=int, default=-1)
    parser.add_argument("--api_num", type=int, default=-1)
    parser.add_argument("--msg", type=str, default="")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--model_path", type=str, default=None)

    args = parser.parse_args()
    print(args)
    model = MODEL_CLASS(model_name=MODEL_NAME)
    qa_test = main(model=model, test_prop=TEST_PROP, few_shot=FEW_SHOT, nearest_k=NEAREST_K)
    with open(TEMP_DATASET, "w", encoding="utf-8") as f:
        for elm in qa_test:
            f.write(json.dumps({key:value for key, value in elm.items() if key != "embedding"}, ensure_ascii=False)+"\n")
    del model
    
    
    test_all(
        model_path=args.model_path,
        model_size=args.model_size,
        dataset=args.dataset,
        steps=args.steps,
        tp=args.tp,
        msg=args.msg,
        max_length=args.max_length,
        demo_num=args.demo_num,
        api_num=args.api_num,
    )
    

"""示例指令

srun --gres=gpu:1 --cpus-per-task=12 -p fnlp-3090 --pty python tester.py \
--model_path=Qwen/Qwen2-7B-Instruct  --msg=test_qwen2_7B_my_template

srun --gres=gpu:1 --cpus-per-task=12 -p fnlp-3090 --pty python tester.py \
--model_path=/remote-home1/share/hf_cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c/  \
    --msg=test_qwen2_7B_my_template

srun --gres=gpu:1 --cpus-per-task=12 -p fnlp-3090 --pty python tester.py \
--model_path=/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B-Instruct  --msg=test_llama_31_8B_my_template_31

srun --gres=gpu:1 --cpus-per-task=12 -p fnlp-3090 --pty python tester.py \
--model_path=akjindal53244/Llama-3.1-Storm-8B  --msg=test_llama_31_8B_my_template_31

srun --gres=gpu:1 --cpus-per-task=12 -p fnlp-3090 --pty python tester.py \
--model_path=Qwen/Qwen2.5-7B-Instruct  --msg=test_qwen25_7B_my_template
"""
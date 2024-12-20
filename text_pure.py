# 导入必要的库
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForMaskedLM, pipeline
from eval_utils import convert_to_tuples, MaskDemaskWrapper  # 确保 eval_utils 正确引用
from datasets import load_dataset

# 主程序执行
if __name__ == '__main__':
    # 设置随机种子以保证结果可复现
    np.random.seed(42)
    torch.cuda.empty_cache()

    # 参数设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测设备
    dataset_val = '100'  # 如果需要测试 1000 条数据，可以改为 '1000'
    defense = 'maj_log'  # 选择防御类型

    # 加载分词器和模型
    ag_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
    ag_model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
    ag_model.to(device)
    print('已加载 AG News 模型和分词器。')

    # 加载微调的 Masked LM 模型
    ag_model_directory = "models/bert-uncased_maskedlm_agnews_july31"
    finetuned_ag_maskedlm = BertForMaskedLM.from_pretrained(ag_model_directory)
    finetuned_ag_maskedlm.to(device)
    ag_fill_mask = pipeline("fill-mask", model=finetuned_ag_maskedlm, tokenizer=ag_tokenizer)
    print('已加载 AG News Masked LM 模型。')

    # 防御参数设置
    num_voter = 11
    mask_pct = 0.3

    # 根据用户输入加载数据集
    if dataset_val == '100':
        loaded_ag_100 = Dataset.load_from_disk('data/filtered_ag_clean_100')
        ag_100 = convert_to_tuples(loaded_ag_100)
        dataset = ag_100
        dataset_name = 'ag-news100'
    elif dataset_val == '1000':
        loaded_ag_1000 = Dataset.load_from_disk('data/filtered_ag_clean_1000')
        ag_1000 = convert_to_tuples(loaded_ag_1000)
        dataset = ag_1000
        dataset_name = 'ag-news1000'
    if dataset_val == 'new_data':
        # 加载 CSV 格式的数据集
        loaded_new_data = load_dataset('csv', data_files='path/to/new_dataset.csv')
        new_data = convert_to_tuples(loaded_new_data['train'])  # 假设列名是 text 和 label
        dataset = new_data
        dataset_name = 'new_dataset'
    else:
        raise ValueError('不支持的样本数量。')

    # 应用选择的防御类型
    if defense == "default":
        ag_wrapper = MaskDemaskWrapper(ag_model, ag_tokenizer, ag_fill_mask, num_voter, mask_pct, 'logit')
    elif defense == "logit":
        ag_wrapper = MaskDemaskWrapper(ag_model, ag_tokenizer, ag_fill_mask, num_voter, mask_pct, 'logit')
    elif defense == 'maj_log':
        ag_wrapper = MaskDemaskWrapper(ag_model, ag_tokenizer, ag_fill_mask, num_voter, mask_pct, 'maj_log')
    elif defense == "one_hot":
        ag_wrapper = MaskDemaskWrapper(ag_model, ag_tokenizer, ag_fill_mask, num_voter, mask_pct, 'maj_one_hot')
    else:
        raise ValueError('无效的防御类型。')

    print(f'使用 num_voter = {num_voter}，mask_pct = {mask_pct}，数据集 = {dataset_name}...')

    # 分类结果输出
    texts, labels = zip(*dataset)  # 解包文本和标签
    predictions = ag_wrapper(texts)  # 调用模型进行分类

    # 打印分类结果
    for i, (text, label, pred) in enumerate(zip(texts, labels, predictions)):
        print(f"样本 {i+1}:")
        print(f"  原始文本: {text}")
        print(f"  原始标签: {label}")
        print(f"  预测结果: {pred}")
        print()

    print(f'分类完成，处理了 {len(dataset)} 个样本。')

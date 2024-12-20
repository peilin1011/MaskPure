from textattack.models.wrappers import ModelWrapper
from transformers import pipeline
import sys
sys.path.append('../')
from utils import *
sys.path.pop()

class MaskDemaskWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, fill_mask, num_voter, mask_pct, v_type):
        self.model = model
        self.tokenizer = tokenizer
        self.fill_mask = fill_mask
        self.num_voter = num_voter
        self.mask_pct = mask_pct
        self.pipeline = pipeline('text-classification', model=model, 
                                 tokenizer=tokenizer, device=next(model.parameters()).device)
        self.v_type = v_type
        
    def __call__(self, text_input_list):
        filled = mask_and_demask(text_input_list, self.tokenizer, self.fill_mask, verbose = False, 
                                 num_voter=self.num_voter, mask_pct=self.mask_pct
                                )
        # get the logits (to inform the attacker)
        logits = get_avg_logits(filled, self.pipeline, num_voter=self.num_voter, v_type=self.v_type)
        outputs = [[value for value in entry.values()] for entry in logits]
        return outputs
    
    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]    
'''

class MaskDemaskWrapperVQA:
    def __init__(self, model, tokenizer, fill_mask, num_voter, mask_pct, v_type, device='cuda:0'):
        """
        VQA 任务的 MaskDemaskWrapper 类。
        
        Parameters:
        - model (torch.nn.Module): 您的 VQA 模型。
        - tokenizer: 分词器（如果需要）。
        - fill_mask: 用于掩码填充的工具或函数。
        - num_voter (int): 每个输入生成的答案数量。
        - mask_pct (float): 掩码比例。
        - v_type (str): 聚合类型，当前方案不需要，可以保留或忽略。
        - device (str): 设备配置，如 'cuda:0' 或 'cpu'。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.fill_mask = fill_mask
        self.num_voter = num_voter
        self.mask_pct = mask_pct
        self.v_type = v_type
        self.device = device
    
    def __call__(self, inputs):
        """
        处理输入并生成最终答案。
        
        Parameters:
        - inputs (list[tuple]): 每个元素是一个包含 (image_tensor, question) 的元组。
        
        Returns:
        - final_answers (list[str]): 每个输入的最终答案。
        """
        filled = self.mask_and_demask(inputs)
        # 获取最终答案
        final_answers = get_avg_answers(filled, self.model, self.num_voter, self.device)
        return final_answers
    
    def mask_and_demask(self, inputs):
        """
        对输入进行掩码和去掩码处理。
        这一步骤取决于您的具体需求和实现方式。
        例如，您可以对问题进行掩码以生成不同版本的问题。
        
        Parameters:
        - inputs (list[tuple]): 每个元素是一个包含 (image_tensor, question) 的元组。
        
        Returns:
        - masked_inputs (list[tuple]): 掩码和去掩码后的输入。
        """
        # 示例实现：假设只对问题进行掩码
        masked_inputs = []
        for image_tensor, question in inputs:
            masked_question = self.fill_mask(question, mask_pct=self.mask_pct)
            #print(f'original_question: {question}')
            #print(f'masked_question: {masked_question}')
            masked_inputs.append((image_tensor, masked_question))
        return masked_inputs
    
    def _tokenize(self, inputs):
        """
        Helper method for tokenization if needed.
        
        Args:
            inputs (list[str]): list of input strings
        
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
'''

def parse_attack_name(attack_name):
    """
    Function to parse attack name from the given attack object.

    Parameters:
    attack_name (object): Attack object

    Returns:
    string: Attack name as string
    """
    return f'{attack_name}'.split('.')[-1].strip("'>")


def convert_to_tuples(data):
    """
    Input data is of type datasets.Dataset
    For example, if you printed the first few of the input, it might look like:
    print(data[:2])
    {'text': ["I enjoyed the movie a lot!", 'Asolutely horrible film'], 'label': [1, 0]}
    
    Returns:
    Dataset in a form that textattack.datasets.Dataset can handle
    """
    tuples = []
    for text, label in zip(data['text'], data['label']):
        tuples.append((text, label))
    return tuples
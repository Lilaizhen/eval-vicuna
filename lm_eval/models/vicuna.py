import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from typing import List, Tuple, Dict

@register_model("vicuna")
class VicunaLM(LM):
    def __init__(self, model_name_or_path="/home/llz/test-lm-eval5/models/vicuna-7b-v1.5", batch_size=1):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).cuda()
        self.batch_size = batch_size
        print(f"Model {model_name_or_path} loaded successfully on GPU.")

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        results = []
        for context, continuation in [req.args for req in requests]:
            print(f"Processing context: {context}, continuation: {continuation}")
            context_enc = self.tok_encode(context, return_tensors='pt').cuda()
            continuation_enc = self.tok_encode(continuation, return_tensors='pt').cuda()

            input_ids = torch.cat([context_enc, continuation_enc], dim=-1)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                log_likelihood = -outputs.loss.item()
                is_greedy = True  # Assuming greedy for simplicity

            results.append((log_likelihood, is_greedy))
        return results

    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        results = []
        for (string,) in [req.args for req in requests]:
            input_ids = self.tok_encode(string, return_tensors='pt').cuda()
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                log_likelihood = -outputs.loss.item()

            results.append((log_likelihood,))
        return results

    def generate_until(self, requests) -> List[str]:
        results = []
        for context, until in [req.args for req in requests]:
            input_ids = self.tok_encode(context, return_tensors='pt').cuda()
            with torch.no_grad():
                generated_ids = self.model.generate(input_ids, max_length=128, eos_token_id=self.tokenizer.encode(until)[0])
                generated_text = self.tok_decode(generated_ids[0])

            results.append(generated_text)
        return results

    @property
    def tokenizer_name(self) -> str:
        return "vicuna"

    @property
    def chat_template(self) -> str:
        return "<s>{role}：{content}</s>"

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        history = ""
        for turn in chat_history:
            history += self.chat_template.format(role=turn['role'], content=turn['content'])
        return history

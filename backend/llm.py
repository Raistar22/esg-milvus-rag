from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

_tokenizer = None
_model     = None
LLM_MODEL  = "google/flan-t5-base"

def get_llm():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        _model     = T5ForConditionalGeneration.from_pretrained(
            LLM_MODEL, dtype=torch.float32
        )
        _model.eval()
    return _tokenizer, _model

def run_llm(prompt, max_new_tokens=512):
    tokenizer, model = get_llm()
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            num_beams=4, early_stopping=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

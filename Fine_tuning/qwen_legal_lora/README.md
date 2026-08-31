---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
language:
- vi
tags:
- base_model:adapter:Qwen/Qwen2.5-3B-Instruct
- lora
- sft
- transformers
- trl
- legal-qa
- vietnamese
license: apache-2.0
---

# Qwen2.5-3B-Instruct Vietnamese Legal QA (LoRA)

Mô hình này là phiên bản tinh chỉnh (fine-tuned) sử dụng phương pháp **LoRA (Low-Rank Adaptation)** trên nền tảng **Qwen/Qwen2.5-3B-Instruct**, chuyên phục vụ tác vụ hỏi đáp và tư vấn thông tin pháp luật Việt Nam (Vietnamese Legal Question Answering).

## Model Details

- **Developed by:** Vietnamese Legal QA Project Team
- **Model type:** Causal Language Model (LoRA Adapter)
- **Language(s):** Tiếng Việt (Vietnamese)
- **License:** Apache-2.0
- **Finetuned from model:** [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Repository:** [Vietnamese-Legal-QA-RAG-FT](https://github.com/thuphamgia/Vietnamese-Legal-QA-RAG-FT)

## Uses

### Direct Use
Mô hình được tối ưu hóa để trả lời câu hỏi, giải thích văn bản quy phạm pháp luật, trích xuất căn cứ pháp lý và hỗ trợ tra cứu các điều luật trong hệ thống pháp luật Việt Nam.

### Out-of-Scope Use
Mô hình không thay thế tư vấn pháp lý chuyên nghiệp từ luật sư hoặc cơ quan nhà nước có thẩm quyền. Không sử dụng cho các mục đích ra quyết định pháp lý ràng buộc hoặc tư vấn trong các vụ án thực tế mà không có sự kiểm chứng của chuyên gia.

## How to Get Started with the Model

Sử dụng thư viện `peft` và `transformers` để tải và suy luận (inference):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-3B-Instruct"
lora_model_path = "./Fine_tuning/qwen_legal_lora"

tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, lora_model_path)

messages = [
    {"role": "system", "content": "Bạn là một trợ lý ảo am hiểu pháp luật Việt Nam. Hãy trả lời câu hỏi một cách chính xác, ngắn gọn và có căn cứ pháp lý."},
    {"role": "user", "content": "Thời hiệu khởi kiện vụ án dân sự được quy định như thế nào?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.9
)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)

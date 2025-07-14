from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型路径，使用原始字符串避免转义问题
model_path = r"D:/modelLib/deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 定义提示并生成回复
prompt = "你好，请问今天天气怎么样？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.85,
    do_sample=True
)

# 解码并打印回复
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"问题: {prompt}")
print(f"回答: {response}")
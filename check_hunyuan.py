from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

model_name_or_path = "tencent/Hunyuan-1.8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")  # You may want to use bfloat16 and/or move to GPU here
messages = [
    {"role": "user", "content": "Write a short summary of the benefits of regular exercise"},
]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,return_tensors="pt",
                                                enable_thinking=True # Toggle thinking mode (default: True)
                                                )
                                                
outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)

output_text = tokenizer.decode(outputs[0])
print("output_text=",output_text)
think_pattern = r'<think>(.*?)</think>'
think_matches = re.findall(think_pattern, output_text, re.DOTALL)

answer_pattern = r'<answer>(.*?)</answer>'
answer_matches = re.findall(answer_pattern, output_text, re.DOTALL)

think_content = [match.strip() for match in think_matches][0]
answer_content = [match.strip() for match in answer_matches][0]
print(f"thinking_content:{think_content}\n\n")
print(f"answer_content:{answer_content}\n\n")

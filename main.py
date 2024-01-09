import json
from llama_cpp import Llama

print("loading model")
llm = Llama(model_path="./models/vicuna-7b-v1.5.Q4_K_M.gguf")
#change model
print("loaded")

prompt = "who u?"
print("running model")
output = llm(
    "Question: {prompt}? Answer:".format(prompt=prompt),
    max_tokens=100,
    stop=["\n","Question:","Q:"],
    echo=True,
)

print(json.dumps(output,indent=2))
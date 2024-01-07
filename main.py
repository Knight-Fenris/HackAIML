import json
from llama_cpp import Llama

print("loading model")
llm = Llama(model_path="./models/pytorch_model-00001-of-00003.bin")
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
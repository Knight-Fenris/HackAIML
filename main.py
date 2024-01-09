import json
from llama_cpp import Llama

# import pandas as pd

# categories = list(pd.read_csv("Categories.csv", nrows=1))
# print(categories)
sentence = 'charity is for rich, I don’t care'


# prompt = "Which of these categoires ({categories})) does {sentence} fall into"
prompt = """Classify the following sentence into the following categories False Urgency,Basket Sneaking,Confirm Shaming,Forced Action,Subscription Trap,Interface interference,Bait and switch,Drip pricing,Disguised advertisement,Nagging, return the probability scores for the sentence, without text
{sentence}
"""
prompt = prompt.format(sentence=sentence)
print(prompt)


print("loading model")
llm = Llama(model_path="./models/vicuna-13b-v1.5-16k.Q4_K_M.gguf")
#change model
print("loaded")


# prompt = "Ill input sentences in the following prompts, return a heatmap in the form of a python list containing numbers between 0 and 1 according to the categories they belong in; the categories being: False Urgency,Basket Sneaking,Confirm Shaming,Forced Action,Subscription Trap,Interface interference,Bait and switch,Drip pricing,Disguised advertisement,Nagging"
# prompt = ""

print("running model")
output = llm(
    "Question: {prompt} Answer:".format(prompt=prompt),
    max_tokens=100,
    stop=["11.","Question:","Q:"],
    echo=True,
)

print(json.dumps(output,indent=2))

# print(llm.create_chat_completion(
#         messages=[{
#          "role": "user",
#          "content": "what is the meaning of life?"
#         }]))

# response = llm.create_chat_completion(
#       messages = [
#         {
#           "role": "user",
#           "content": "Extract Jason is 25 years old"
#         }
#       ]
# )

# print(response)

import json
from llama_cpp import Llama

# import pandas as pd

# categories = list(pd.read_csv("Categories.csv", nrows=1))
# print(categories)
sentence = 'sale 2 left! Hurry '


# prompt = "Which of these categoires ({categories})) does {sentence} fall into"
prompt = """Classify the Input into these categories: False Urgency,Basket Sneaking,Confirm Shaming,Forced Action,Subscription Trap,Interface interference,Bait and switch,Drip pricing,Disguised advertisement,Nagging; by returning only 10 probability scores respectively; no text"""
print(prompt)

file1 = open('text_out.txt', 'r')
lines = file1.readlines()
file1.close()

file1 = open("patterns.txt", "w")



print("loading model")  
llm = Llama(model_path="./models/vicuna-13b-v1.5.Q6_K.gguf")
#change model
print("loaded")


print("running model")

import re

for i in lines:
    sentence = i
    output = llm(
        "Instruction: {prompt} \nInput:{sentence} \nOutput: ".format(prompt=prompt,sentence=sentence),
        max_tokens=100,
        stop=["11.","Question:","Q:", "Instruction:", "Input:", "Output:"],
        echo=True,
        temperature=0.8,
        top_p=0.8,

    )

    # ans = json.loads(json.dumps(output,indent=2))
    #for new model run first command only then format
    print(sentence)

    ans = output["choices"][0]['text']
    ans = ans.split("Output:",1)[1]
    ans = ans.replace("\r", " ").replace("\n", " ")
    # non_d = re.compile(r'[^\d.]+')
    ans = re.sub(r'[^\d. ]+', '', ans)
    ans = re.sub(' +', ' ', ans)
    ans = ans.strip()
    print(ans)
    file1.write(ans + '\n')

file1.close()
# print(ans)

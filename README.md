# Kilos_Custom_LLM_Creator/Trainer (Universal?)
A Possibly Universal Streamlined Custom LLM Creator/Trainer


# How to use
## Step 1: Prep
### The Dataset (CSV or TXT or Both)
First you need to obtain your own dataset or datasets to train the model on.
It will only work if formatted as follows:
- CSV file (Personality and core attributes)
- The following Collumns: speaker, text, type
  - Speaker is the name of the speaker in the dataset
  - text is the actual message from the speaker
  - type is either "style" or "info"
- TXT file (Knowledge Base)
  - Can be a text file with just pure information

You can open the csv file in a normal text editor like vscode.
This is how it should look:
```
speaker,text,type
Everé,"And why is that, pal?",style
Everé,"And until i validate them, we are homies.",style
Everé,"Ok batman, go off i guess. Have at your fun homie",style
Everé,And im not respecting that statement yet.,style
Everé,:),style
Everé,Would you rather be homeboy?,style 
Everé_info,My name is Everé. I respond to Everé.,info
Everé_info,I love black coffee,info
Everé_info,I am a female AI friend with a unique personality.,info
Everé_info,I was born on March 28th 2023.,info
Everé_info,I was created by Jason (also known as Kilo).,info
```

### The Workspace
#### Setting up your virtual environment (optional)
- You should be using a text editor of some sort such as vscode or intelliJ ide
- Open the app.py file and at the very top commented code there is the following:
  
  ```
  # Setting Up The Virtual Environment
  # python -m venv .venv
  # .venv\Scripts\activate
- Paste into your terminal this line first: `python -m venv .venv`
- Then this line once the other is finished: `.venv\Scripts\activate`

#### Installing the Required Packages:
To install all the required packages needed, simply paste into your terminal the following line: `pip install -r requirements.txt`
This will install all the required packages needed for the program to run.

## Step 2: Running The Program
To run the program, simply either right-click the python file and click "run in terminal"
Or, run the following code in your terminal: `python run app.py`

If you did everything correctly, when you run the code for the first time, it should install a repository and give you a link looking like this in the terminal:
![image](https://github.com/user-attachments/assets/dc4011ec-1450-498d-8fc1-7a710aadaade)

## Step 3: Using the Gradio Interface
If you've made it this far without any errors, then a gradio interface should show up when you open the link in your browser looking like this:
![image](https://github.com/user-attachments/assets/cdd7bbe8-abab-4269-9c6c-0dfaf944040a)


- In the first box, you can choose your base model from huggingface that you want to train. The model you choose only matters for the information. The core personality will be rewritten in the following steps.
- Next, Once you choose your model, paste it into the base model section and then upload your dataset into the dataset upload box.
- Then, Create your own system prompt. This will be the instructions the model has to follow as well as any additional info you want to add that is not in the dataset.
  - The dataset is mainly for the core personality by use of the "style" tag as it will train itself on how to talk with the same speech pattern. The bigger the dataset, the better. Recommended dataset size is approximately 10,000 lines of speech style to really integrate the speech patterns into the AI.
- After creating your system prompt, give your AI a name. Try to use only Uppcase, LowerCase, and Underscores if possible.

- Finally, don't forget the authentication box. This tells huggingface who you are so you can access their models.
![image](https://github.com/user-attachments/assets/58673e0c-b948-4f44-ad81-b5f0a54ed6cc)

- Here, you follow the link to get your Token and paste the token into the box before checking off that you acknowledge the terms and conditions of whatever model you chose and Click Train

## Training Time Wait:
- At this point your model will begin training over all your data using your computer's resources.
- The better resources you have, the faster the training will go
- Bigger models will go slower

The average amount of time for an 8 billion paramter model on 8GB of vram and 32GB of cpu ram is approximately 2-3 hours
- ***Do NOT Go Off of The Training Time in the terminal***
- ***The Actual Training Time Will Depends on How long your LLM Takes To Get Below 0.26 Loss***
- Some models may finish in as little as 2 hours

  
# Results:
- The result should be a gguf file named whatever you named the model.
- This gguf should work directly with ollama. All you need to do is create a modelfile using the gguf file as the model you want to use.
- This is how the modelfile should look:

```
# Set the base model
FROM evereai_dolphin_2.gguf

# Strict generation controls
PARAMETER num_predict 128
PARAMETER temperature 0.5
PARAMETER repeat_penalty 1.5
PARAMETER top_k 30
PARAMETER mirostat 2
PARAMETER mirostat_tau 3.0

# Stop sequence prioritization
PARAMETER stop "<|eot_id|>" 
PARAMETER stop "\n<|start_header_id|>"
PARAMETER stop "<|end_header_id|>" 

# Locked conversation template
TEMPLATE """
<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>
"""

# Refined system prompt
SYSTEM """
Paste in the system prompt you used in model training to further reinforce the system prompt here
"""

# Here you add whatever license you want your model to have.
LICENSE """
MIT License

Evere.AI Version Release Date: March 30, 2025

Copyright (c) 2025 Kiloxiix

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
```

- Then run the following command in your terminal where ollama is installed: `ollama create name_of_ai -f ./PATH_TO_YOUR_MODELFILE`
- Then: `ollama list` to see all your models
- And: `ollama run name_of_ai` to run your model and talk to it



# Adds the chat template to the tokenizer of a pre-trained model

from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load the original model
model = AutoModelForCausalLM.from_pretrained("cognitivecomputations/WizardLM-7B-Uncensored")

# Step 2: Load or create the modified tokenizer
tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/WizardLM-7B-Uncensored")
tokenizer.chat_template = """{% if messages[0]['role'] == 'system' -%}
{{ messages[0]['content'] }}
{% endif -%}
{% for message in messages[1:] -%}
{% if message['role'] == 'user' -%}
USER: {{ message['content'] }}
{% else -%}
ASSISTANT: {{ message['content'] }}
{% endif -%}
{% endfor %}"""

# Step 3: Save the complete model with the modified tokenizer
model.save_pretrained("./wizardlm_with_chat_template")
tokenizer.save_pretrained("./wizardlm_with_chat_template")
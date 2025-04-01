# Setting Up The Virtual Environment
# python -m venv .venv
# .venv\Scripts\activate



import os
import gradio as gr
import torch
import subprocess
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from huggingface_hub import login, HfFolder, model_info, ModelCard
import pandas as pd
import tempfile

# Check for llama.cpp installation
if not os.path.exists("llama.cpp"):
    os.system("git clone https://github.com/ggerganov/llama.cpp.git")

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold=0.3):
        self.threshold = threshold
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and 'loss' in state.log_history[-1]:
            current_loss = state.log_history[-1]['loss']
            if current_loss < self.threshold:
                control.should_training_stop = True
        return control

def display_error(message):
    """Helper function to display errors in Gradio"""
    raise gr.Error(message)

def check_model_requirements(model_id, agree_to_terms, hf_token):
    """Verify model requirements and user authentication"""
    try:
        # Check if we have a token for gated models
        if hf_token:
            HfFolder.save_token(hf_token)
            
        info = model_info(model_id, token=hf_token)
        
        if info.gated:
            if not agree_to_terms:
                license_link = f"https://huggingface.co/{model_id}"
                display_error(f"⚠️ This model requires accepting its terms.\n"
                             f"1. Visit {license_link}\n"
                             f"2. Accept the license\n"
                             f"3. Check the agreement box below")
                
            # Verify token is present and valid
            if not hf_token:
                display_error("🔑 This model requires Hugging Face authentication.\n"
                              "Please provide your HF API token in the 'HF Token' field")
                
            # Verify actual access
            try:
                ModelCard.load(model_id, token=hf_token)
            except Exception as e:
                display_error(f"❌ Authentication failed: {str(e)}\n"
                              "Ensure you've:\n"
                              "1. Accepted the model's terms on Hugging Face\n"
                              "2. Used a valid access token")
                
    except Exception as e:
        if "404" in str(e):
            display_error(f"Model {model_id} not found on Hugging Face Hub")
        elif "401" in str(e):
            display_error("🔒 Invalid or missing Hugging Face token\n"
                          "Get your token from: https://huggingface.co/settings/tokens")
        else:
            display_error(f"License verification failed: {str(e)}")
            
    return True

def prepare_dataset(csv_path, system_prompt):
    """Convert CSV to conversational format with custom system prompt"""
    try:
        df = pd.read_csv(csv_path)
        conversations = []
        
        for _, row in df.iterrows():
            clean_response = row['text'].split("-éveré")[0].strip()
            if len(clean_response.split()) > 100:
                clean_response = ' '.join(clean_response.split()[:40]) + "..."
            clean_response += " <|eot_id|>"
            
            if row['type'] == 'style':
                text = f"""<|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Generate a response<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{clean_response}"""
            elif row['type'] == 'info':
                text = f"""<|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Tell me about yourself<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{clean_response}"""
            
            conversations.append({"text": text})
        
        return Dataset.from_pandas(pd.DataFrame(conversations))
    except Exception as e:
        display_error(f"Dataset preparation failed: {str(e)}")

def train_model(base_model, dataset_path, system_prompt, output_dir, hf_token):
    """Handle model training with progress updates"""
    try:
        # Check hardware configuration
        use_cpu = not torch.cuda.is_available()
        
        if use_cpu:
            print("⚠️ No GPU detected - Falling back to CPU mode (much slower)")
            
            # Display warning but continue with CPU
            gr.Warning("No GPU detected - Training on CPU (much slower). "
                     "For better performance use a GPU-enabled environment.")

        # Configure quantization based on hardware
        if use_cpu:
            # CPU-compatible configuration
            bnb_config = None
            torch_dtype = torch.float32
            device_map = None  # Let transformers handle device placement
        else:
            # GPU-optimized configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            torch_dtype = torch.float16
            device_map = "auto"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            legacy=False, 
            padding_side='right', 
            token=hf_token
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({
            'additional_special_tokens': [
                '<|start_header_id|>', 
                '<|end_header_id|>', 
                '<|eot_id|>'
            ]
        })

        # Prepare dataset
        dataset = prepare_dataset(dataset_path, system_prompt)
        
        # Load model with appropriate configuration
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=hf_token,
            low_cpu_mem_usage=not use_cpu  # Only enable for GPU
        )
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model) if not use_cpu else model
        model = get_peft_model(model, LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        ))
        
        model.resize_token_embeddings(len(tokenizer))

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda examples: tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=384,
                return_tensors='pt'
            ),
            batched=True,
            remove_columns=['text']
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2 if not use_cpu else 1,  # Smaller batch for CPU
            gradient_accumulation_steps=16 if not use_cpu else 8,  # Fewer steps for CPU
            learning_rate=2e-5,
            fp16=not use_cpu,  # Disable mixed precision for CPU
            bf16=use_cpu,  # Use bfloat16 if on CPU
            max_grad_norm=0.3,
            max_steps=2000,
            logging_steps=10,
            save_steps=500,
            gradient_checkpointing=not use_cpu,  # Disable for CPU
            optim="adamw_torch",
            report_to="none",
            disable_tqdm=use_cpu  # Disable progress bars for CPU (often slower)
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[EarlyStoppingCallback(0.25)]
        )
        
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
    except Exception as e:
        display_error(f"Model training failed: {str(e)}")

def convert_to_gguf(merged_model_path, gguf_path):
    """Convert merged model to GGUF format"""
    try:
        subprocess.run([
            "python",
            "llama.cpp/convert_hf_to_gguf.py",
            merged_model_path,
            "--outfile",
            gguf_path,
            "--outtype",
            "f16"
        ], check=True)
    except subprocess.CalledProcessError as e:
        display_error(f"GGUF conversion failed: {str(e)}")
    except Exception as e:
        display_error(f"Unexpected error during GGUF conversion: {str(e)}")

def full_training_pipeline(base_model, dataset_file, system_prompt, model_name, agree_to_terms, hf_token):
    """End-to-end training pipeline with error handling"""
    try:
        # Verify model requirements first
        check_model_requirements(base_model, agree_to_terms, hf_token)
        
        # Authenticate with HF token if provided
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)
            
        # Create temporary paths
        output_dir = tempfile.mkdtemp()
        merged_dir = tempfile.mkdtemp()
        gguf_path = f"/tmp/{model_name}.gguf"

        # 1. Fine-tune model
        train_model(base_model, dataset_file.name, system_prompt, output_dir, hf_token)

        # 2. Merge model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            token=hf_token
        ).to('cuda')
        model = PeftModel.from_pretrained(base_model, output_dir)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        tokenizer.save_pretrained(merged_dir)

        # 3. Convert to GGUF
        convert_to_gguf(merged_dir, gguf_path)

        return gguf_path
    except Exception as e:
        display_error(f"Training pipeline failed: {str(e)}")

with gr.Blocks() as demo:
    gr.Markdown("# 🦙 Kilo's Custom LLM Creator/Trainer")
    
    with gr.Row():
        with gr.Column():
            base_model = gr.Textbox(
                label="Base Model",
                value="cognitivecomputations/Dolphin3.0-Llama3.1-8B"
            )
            dataset = gr.File(label="Training Dataset (CSV)", file_types=[".csv"])
            system_prompt = gr.Textbox(
                label="System Prompt",
                lines=8,
                value="You are an AI assistant...",
                interactive=True
            )
            model_name = gr.Textbox(label="Output Model Name", value="my-custom-model")
            with gr.Accordion("🔐 Authentication", open=False):
                gr.Markdown("Required for gated models like Gemma or Llama-2\n"
                           "[Get your HF token](https://huggingface.co/settings/tokens)")
                hf_token = gr.Textbox(
                    label="Hugging Face Access Token",
                    type="password",
                    placeholder="hf_XXXXXXXXXXXXXX"
                )
                agree_to_terms = gr.Checkbox(
                    label="I acknowledge and comply with the model's license terms",
                    info="Required for models with special licenses"
                )
            train_btn = gr.Button("🚀 Start Training")
        
        with gr.Column():
            output_file = gr.File(label="Download GGUF Model")
            status = gr.Markdown()

    train_btn.click(
        fn=full_training_pipeline,
        inputs=[base_model, dataset, system_prompt, model_name, agree_to_terms, hf_token],
        outputs=output_file,
        api_name="train"
    )

demo.launch()
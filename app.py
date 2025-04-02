# Setting Up The Virtual Environment
# python -m venv .venv
# .venv\Scripts\activate


import os
import gradio as gr
import torch
import subprocess
import time
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
import sys
import platform

# Add GPU diagnostics
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {prop.name} with {prop.total_memory / 1e9:.2f} GB memory")
else:
    print("No GPU detected. Training will run on CPU (much slower).")
print("=" * 50)

# Check for llama.cpp installation
if not os.path.exists("llama.cpp"):
    print("Cloning llama.cpp repository...")
    os.system("git clone https://github.com/ggerganov/llama.cpp.git")

# Global state variables for progress tracking
current_step = 0
total_steps = 0
current_stage = "Initializing"
overall_progress = 0.0

class ProgressCallback(TrainerCallback):
    def __init__(self, progress_fn, status_md, loss_chart):
        self.progress_fn = progress_fn
        self.status_md = status_md
        self.loss_chart = loss_chart
        self.loss_history = []
        self.step_history = []
        self.current_loss = None  # Track loss as instance variable
        
    def on_train_begin(self, args, state, control, **kwargs):
        global current_stage, total_steps, current_step
        current_stage = "Training"
        total_steps = args.max_steps
        current_step = 0
        print(f"Batch size: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        if torch.cuda.is_available():
            print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB free")
        
    def on_step_end(self, args, state, control, **kwargs):
        global current_step, overall_progress
        current_step = state.global_step
        
        # Update progress
        step_progress = min(current_step / total_steps, 1.0)
        overall_progress = 0.3 + (step_progress * 0.4)  # 30-70% range for training
        
        # Update UI components
        if self.progress_fn:
            try:
                self.progress_fn(overall_progress, desc="Training Progress")
            except Exception as e:
                print(f"Warning: Failed to update progress: {str(e)}")
        
        # Update loss tracking
        if state.log_history and len(state.log_history) > 0:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.current_loss = latest_log['loss']  # Store loss as instance variable
                self.loss_history.append(self.current_loss)
                self.step_history.append(current_step)
                
                # Update loss chart
                try:
                    new_plot = {
                        "data": [{"x": self.step_history, "y": self.loss_history, "type": "line"}],
                        "layout": {"title": "Training Loss", "xaxis": {"title": "Steps"}, "yaxis": {"title": "Loss"}}
                    }
                    self.loss_chart.value = new_plot
                except Exception as e:
                    print(f"Warning: Failed to update loss chart: {str(e)}")
                
                # Update status message
                try:
                    status_text = f"**Training Model**: Step {current_step}/{total_steps} | Loss: {self.current_loss:.4f}"
                    self.status_md.value = status_text
                except Exception as e:
                    print(f"Warning: Failed to update status: {str(e)}")
                
                # Early stopping check
                if self.current_loss <= 0.26:
                    control.should_training_stop = True
                    try:
                        self.status_md.value = f"**Early stopping triggered**: Loss {self.current_loss:.4f} below threshold 0.26"
                    except Exception:
                        pass
        
        return control
        
    def on_train_end(self, args, state, control, **kwargs):
        global current_stage
        current_stage = "Model trained successfully"
        try:
            # Get final loss either from last log or our tracked value
            final_loss = None
            if state.log_history:
                for log in reversed(state.log_history):
                    if 'loss' in log:
                        final_loss = log['loss']
                        break
            
            # Fallback to our tracked loss if not found in logs
            if final_loss is None:
                final_loss = self.current_loss
            
            if final_loss is not None:
                self.status_md.value = f"**Training completed**: Final loss {final_loss:.4f}"
            else:
                self.status_md.value = "**Training completed** (loss not recorded)"
        except Exception as e:
            print(f"Warning: Failed to update final status: {str(e)}")
            self.status_md.value = "**Training completed**"

def display_error(message, status_md=None):
    """Helper function to display errors in Gradio"""
    print(f"ERROR: {message}")
    if status_md is not None:
        try:
            status_md.value = f"❌ **Error**: {message}"
        except Exception:
            pass
    raise gr.Error(message)

def update_progress(progress_fn, status_md, stage, progress_value, desc="Processing"):
    """Update progress indicators"""
    global current_stage, overall_progress
    current_stage = stage
    overall_progress = progress_value
    
    if progress_fn is not None:
        try:
            progress_fn(progress_value, desc=desc)
        except Exception as e:
            print(f"Warning: Could not update progress bar: {str(e)}")
    
    if status_md is not None:
        try:
            status_md.value = f"**{stage}**"
        except Exception as e:
            print(f"Warning: Could not update status: {str(e)}")

def check_model_requirements(model_id, agree_to_terms, hf_token, progress_fn, status_md):
    """Verify model requirements, authentication, and template support"""
    try:
        update_progress(progress_fn, status_md, f"Checking model requirements for {model_id}", 0.05, "Verifying Model")
        
        # Authentication checks
        if hf_token:
            HfFolder.save_token(hf_token)
            
        # Get model info
        info = model_info(model_id, token=hf_token)
        
        # Check for gated model
        if info.gated:
            if not agree_to_terms:
                license_link = f"https://huggingface.co/{model_id}"
                display_error(f"⚠️ This model requires accepting its terms.\n"
                             f"1. Visit {license_link}\n"
                             f"2. Accept the license\n"
                             f"3. Check the agreement box below", status_md)
                
            if not hf_token:
                display_error("🔑 This model requires Hugging Face authentication.\n"
                              "Please provide your HF API token in the 'HF Token' field", status_md)
                
            try:
                ModelCard.load(model_id, token=hf_token)
            except Exception as e:
                display_error(f"❌ Authentication failed: {str(e)}\n"
                              "Ensure you've:\n"
                              "1. Accepted the model's terms on Hugging Face\n"
                              "2. Used a valid access token", status_md)

        # Check template requirements
        update_progress(progress_fn, status_md, "Checking chat template support", 0.08, "Template Check")
        
        # Load tokenizer to check template
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            legacy=False,
            padding_side='right'
        )
        
        # Verify chat template support
        if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
            display_error(f"Model {model_id} doesn't support chat templates.\n"
                          "Please use a model with configured chat_template in tokenizer.", 
                          status_md)
            
        # Verify essential special tokens
        required_tokens = {'eos_token', 'bos_token', 'pad_token'}
        missing_tokens = [tok for tok in required_tokens if getattr(tokenizer, tok) is None]
        if missing_tokens:
            display_error(f"Missing required tokens: {', '.join(missing_tokens)}\n"
                          "Please use a model with properly configured special tokens.",
                          status_md)

        update_progress(progress_fn, status_md, "Model requirements check passed!", 0.1, "Verification Complete")
        return True

    except Exception as e:
        if "404" in str(e):
            display_error(f"Model {model_id} not found on Hugging Face Hub", status_md)
        elif "401" in str(e):
            display_error("🔒 Invalid or missing Hugging Face token\n"
                          "Get your token from: https://huggingface.co/settings/tokens", status_md)
        else:
            display_error(f"Verification failed: {str(e)}", status_md)

def train_model(base_model, dataset_path, system_prompt, output_dir, hf_token, progress_fn, status_md, loss_chart):
    """Universal model training with dynamic template handling"""
    try:
        torch.cuda.empty_cache()
        use_cpu = not torch.cuda.is_available()
        
        # Hardware configuration
        if use_cpu:
            update_progress(progress_fn, status_md, "⚠️ No GPU detected - Using CPU (slow)", 0.21, "Hardware Check")
            gr.Warning("No GPU detected - Training on CPU (much slower). For better performance use a GPU-enabled environment.")
        else:
            update_progress(progress_fn, status_md, f"🖥️ Training on GPU: {torch.cuda.get_device_name(0)}", 0.21, "Hardware Check")

        # Configure quantization
        bnb_config = None if use_cpu else BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        torch_dtype = torch.float32 if use_cpu else torch.float16
        device_map = None if use_cpu else "auto"

        # Load tokenizer with template validation
        update_progress(progress_fn, status_md, f"Loading tokenizer from {base_model}", 0.23, "Loading Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            legacy=False, 
            padding_side='right', 
            token=hf_token
        )
        
        # Configure special tokens
        special_tokens = {}
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            display_error("Model does not have a chat template configured", status_md)

        # Dataset preparation with dynamic templating
        update_progress(progress_fn, status_md, "Preparing dataset with model's template", 0.24, "Dataset Setup")
        def prepare_dataset(csv_path, system_prompt, tokenizer):
            df = pd.read_csv(csv_path)
            conversations = []
            
            for _, row in df.iterrows():
                clean_response = row['text'].split("-éveré")[0].strip() if "-éveré" in row['text'] else row['text'].strip()
                
                # Create message structure
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate a response" if row['type'] == 'style' else "Tell me about yourself"},
                    {"role": "assistant", "content": clean_response}
                ]
                
                # Apply model's chat template
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                conversations.append({"text": text})
            
            return Dataset.from_pandas(pd.DataFrame(conversations))
        
        dataset = prepare_dataset(dataset_path, system_prompt, tokenizer)

        # Model loading
        update_progress(progress_fn, status_md, f"Loading model from {base_model}", 0.25, "Loading Model")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=hf_token,
            low_cpu_mem_usage=not use_cpu
        )

        # Prepare for training
        update_progress(progress_fn, status_md, "Preparing model for training", 0.27, "Setup")
        if not use_cpu:
            model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        # Tokenization function
        update_progress(progress_fn, status_md, "Tokenizing dataset", 0.29, "Preprocessing")
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=384,
                return_tensors='pt'
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        # Training configuration
        update_progress(progress_fn, status_md, "Setting up training arguments", 0.3, "Configuration")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2 if not use_cpu else 1,
            gradient_accumulation_steps=16 if not use_cpu else 8,
            learning_rate=2e-5,
            fp16=not use_cpu,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=2000,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            gradient_checkpointing=not use_cpu,
            optim="adamw_torch",
            report_to="none",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            warmup_steps=50,
            weight_decay=0.01
        )

        # Training execution
        progress_callback = ProgressCallback(progress_fn, status_md, loss_chart)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[progress_callback]
        )

        update_progress(progress_fn, status_md, "Starting training", 0.3, "Training Model")
        trainer.train()
        
        # Model saving
        update_progress(progress_fn, status_md, f"Saving trained model", 0.7, "Saving")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        update_progress(progress_fn, status_md, "Model training completed!", 0.7, "Training Complete")
        
    except Exception as e:
        display_error(f"Model training failed: {str(e)}", status_md)

        

def convert_to_gguf(merged_model_path, gguf_path, progress_fn, status_md):
    """Convert merged model to GGUF format with proper Windows path handling"""
    try:
        update_progress(progress_fn, status_md, "Checking dependencies for GGUF conversion", 0.8, "Dependency Check")
        
        # Make sure required packages are installed
        required_packages = ["transformers", "numpy", "sentencepiece", "protobuf"]
        for package in required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
                print(f"Installed or confirmed {package}")
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to install {package}, conversion may fail")
        
        # Handle llama.cpp path
        llama_script = os.path.join("llama.cpp", "convert_hf_to_gguf.py")
        if not os.path.exists(llama_script):
            display_error(f"Could not find conversion script at {llama_script}", status_md)
            return None, f"Missing conversion script: {llama_script}"
            
        update_progress(progress_fn, status_md, f"Converting model to GGUF format", 0.85, "GGUF Conversion")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(gguf_path)), exist_ok=True)
        
        # Print paths for debugging
        print(f"Merged model path: {merged_model_path}")
        print(f"GGUF output path: {gguf_path}")
        
        # Run the conversion process with full command logging
        cmd = [
            sys.executable, 
            llama_script,
            merged_model_path,
            "--outfile", gguf_path,
            "--outtype", "f16",
            "--vocab-type", "bpe"  # Critical fix for Llama-style models
        ]
        print(f"Running conversion command: {' '.join(cmd)}")
        
        # Execute the conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Log command output regardless of success
        print(f"Command stdout: {result.stdout}")
        print(f"Command stderr: {result.stderr}")
        
        if result.returncode != 0:
            # If conversion failed, try to extract useful error information
            error_details = result.stderr if result.stderr else "No error details available"
            
            # Implement a fallback: just save the model in Hugging Face format
            fallback_dir = os.path.join(os.path.dirname(gguf_path), f"{os.path.basename(gguf_path)}_hf_format")
            os.makedirs(fallback_dir, exist_ok=True)
            
            # Copy the merged model files to the fallback directory
            import shutil
            shutil.copytree(merged_model_path, fallback_dir, dirs_exist_ok=True)
            
            return fallback_dir, f"GGUF conversion failed. Model saved in Hugging Face format at {fallback_dir}. Error: {error_details}"
        
        # Verify the output file was created
        if not os.path.exists(gguf_path):
            return None, f"Conversion appeared to succeed but output file not found at {gguf_path}"
            
        update_progress(progress_fn, status_md, f"GGUF conversion completed", 0.9, "Conversion Complete")
        return gguf_path, None
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Conversion error: {error_details}")
        return None, f"Error during GGUF conversion: {str(e)}"

def full_training_pipeline(base_model, dataset_file, system_prompt, model_name, agree_to_terms, hf_token, progress=None):
    """End-to-end training pipeline with error handling"""
    status_md = gr.Markdown(value="**Starting pipeline...**")
    loss_chart = gr.Plot()
    
    try:
        def safe_progress(value, desc="Processing"):
            try:
                if progress is not None:
                    progress(value, desc=desc)
            except Exception as e:
                print(f"Progress update failed: {str(e)}")
        
        status_md.value = f"**Starting training pipeline for {model_name}**"
        safe_progress(0.01, "Initialization")
        
        check_model_requirements(base_model, agree_to_terms, hf_token, safe_progress, status_md)
        
        if hf_token:
            status_md.value = "**Authenticating with Hugging Face**"
            safe_progress(0.12, "Authentication")
            login(token=hf_token, add_to_git_credential=False)
            
        # Create platform-compatible paths
        output_dir = tempfile.mkdtemp()
        merged_dir = tempfile.mkdtemp()
        
        # Use tempfile.gettempdir() instead of hardcoded /tmp/ for cross-platform compatibility
        gguf_filename = f"{model_name}.gguf"
        gguf_path = os.path.join(tempfile.gettempdir(), gguf_filename)
        
        status_md.value = f"**Temporary directories created**"
        safe_progress(0.13, "Setup")

        # 1. Fine-tune model
        status_md.value = "**Step 1: Preparing for fine-tuning**"
        safe_progress(0.14, "Training Preparation")
        train_model(base_model, dataset_file, system_prompt, output_dir, hf_token, safe_progress, status_md, loss_chart)

        # 2. Merge model with GPU optimization and safe fallback
        status_md.value = "**Step 2: Merging model**"
        safe_progress(0.71, "Model Merging")
        
        try:
            with open(os.path.join(output_dir, "original_vocab_size.txt"), "r") as f:
                original_vocab_size = int(f.read().strip())
                print(f"Retrieved original vocabulary size: {original_vocab_size}")
        except Exception as e:
            print(f"Warning: Could not read original vocab size: {str(e)}")
            original_vocab_size = None
        
        # Load tokenizer first
        status_md.value = "**Loading fine-tuned tokenizer**"
        safe_progress(0.715, "Loading Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Try loading base model on GPU first
        status_md.value = f"**Loading base model with GPU optimization**"
        safe_progress(0.72, "Loading Model")
        
        try:
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                token=hf_token,
                device_map="auto",
                max_memory={0: "7GB"}  # Reserve 1GB for system
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # Fallback to CPU offloading
                status_md.value = "⚠️ **GPU memory full - Offloading to CPU**"
                safe_progress(0.72, "Optimizing Memory")
                
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float32,
                    token=hf_token,
                    device_map="auto",
                    offload_folder="offload",
                    offload_state_dict=True,
                    max_memory={0: "5GB", "cpu": "16GB"}  # More aggressive offloading
                )
            else:
                raise e

        # Resize embeddings
        status_md.value = "**Resizing base model token embeddings**"
        safe_progress(0.73, "Preparing Model")
        base_model_obj.resize_token_embeddings(len(tokenizer))
        
        # Load adapter with same device mapping
        status_md.value = "**Loading adapter**"
        safe_progress(0.74, "Loading Adapter")
        model = PeftModel.from_pretrained(
            base_model_obj, 
            output_dir,
            device_map=base_model_obj.device_map if hasattr(base_model_obj, 'device_map') else None
        )

        # Merge with memory monitoring
        status_md.value = "**Merging adapter with base model**"
        safe_progress(0.75, "Merging")
        
        try:
            merged_model = model.merge_and_unload()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                status_md.value = "⚠️ **GPU memory full - Retrying merge on CPU**"
                safe_progress(0.75, "Memory Optimization")
                
                model = model.to('cpu')
                base_model_obj = base_model_obj.to('cpu')
                merged_model = model.merge_and_unload()
            else:
                raise e

        # Save the merged model
        status_md.value = f"**Saving merged model**"
        safe_progress(0.77, "Saving")
        
        if not str(next(merged_model.parameters()).device).startswith('cpu'):
            merged_model = merged_model.to('cpu')
            
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        # 3. Convert to GGUF
        status_md.value = "**Step 3: Converting to GGUF format**"
        safe_progress(0.8, "GGUF Conversion")
        gguf_result, error_msg = convert_to_gguf(merged_dir, gguf_path, safe_progress, status_md)

        # Update the gguf_path when returning it at the end
        if os.path.exists(gguf_path):
            status_md.value = f"**✅ Pipeline completed successfully!**"
            safe_progress(1.0, "Complete")
            return gr.File(value=gguf_path, visible=True), status_md, loss_chart
        elif os.path.exists(merged_dir):
            # Create a zip of the merged model directory
            import shutil
            zip_path = shutil.make_archive(merged_dir, 'zip', merged_dir)
            status_md.value = f"**⚠️ GGUF conversion failed - Downloading model in Hugging Face format**"
            safe_progress(1.0, "Complete")
            return gr.File(value=zip_path, visible=True), status_md, loss_chart
        else:
            status_md.value = f"**❌ Pipeline failed - No output files generated**"
            return None, status_md, loss_chart
            
    except Exception as e:
        error_msg = f"Training pipeline failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        if status_md is not None:
            status_md.value = f"❌ **Error**: {error_msg}"
        return None, status_md, loss_chart

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
            output_file = gr.File(label="Download GGUF Model", visible=False)
            gpu_info = gr.Markdown()
            status_md_placeholder = gr.Markdown("**Ready to start training**")
            loss_chart_placeholder = gr.Plot(label="Training Loss")
            
            if torch.cuda.is_available():
                gpu_info.value = f"🖥️ **GPU Detected:** {torch.cuda.get_device_name(0)}\n" + \
                              f"**Memory:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n" + \
                              f"**CUDA Version:** {torch.version.cuda}"
            else:
                gpu_info.value = "⚠️ **No GPU Detected**\n" + \
                              "Training will run on CPU (significantly slower).\n" + \
                              "For better performance, use a system with an NVIDIA GPU."

    train_btn.click(
        fn=full_training_pipeline,
        inputs=[base_model, dataset, system_prompt, model_name, agree_to_terms, hf_token],
        outputs=[output_file, status_md_placeholder, loss_chart_placeholder],
        api_name="train"
    ).then(
        lambda: gr.File(visible=True),
        outputs=output_file
    )

demo.launch()

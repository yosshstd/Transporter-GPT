from unsloth import FastLanguageModel
import torch
import json
from build_instruction_dataset import DataCollatorForSupervisedDataset
from huggingface_hub import login
from datasets import Dataset
import wandb
import omegaconf
import os
from dotenv import load_dotenv
from unsloth import UnslothTrainer, UnslothTrainingArguments
from trl import SFTTrainer

def main():
    # Load configuration
    config = omegaconf.OmegaConf.load("config.yaml")
    
    # Load environment variables
    load_dotenv()
    
    print("Initializing...")
    # Initialize WandB
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    # Authenticate HuggingFace
    login(os.getenv("HUGGINGFACE_TOKEN"))

    print("Loading model and tokenizer...")
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(**config.model)
    
    # Configure LoRA parameters
    model = FastLanguageModel.get_peft_model(model, **config.lora)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    print("Loading dataset...")
    # Load dataset
    train_dataset = Dataset.from_parquet(config.dataset.train_path)
    #test_dataset = Dataset.from_parquet(config.dataset.test_path)
    print(f"Loaded train_dataset size: {len(train_dataset)}")
    #print(f"Loaded test_dataset size: {len(test_dataset)}")
    
    # Configure training arguments
    training_args = UnslothTrainingArguments(**config.training)
    
    # Initialize trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        #eval_dataset = test_dataset,
        data_collator = data_collator,
        args = training_args
    )
    
    print("Training...")
    # Start training
    trainer.train(
        #resume_from_checkpoint=True,
    )

if __name__ == "__main__":
    main()
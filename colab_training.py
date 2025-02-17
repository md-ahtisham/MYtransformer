# Step 1: Clone the GitHub repository
!git clone https://github.com/md-ahtisham/MYtransformer.git

# Step 2: Change directory to the cloned repository
import os
os.chdir('/content/MYtransformer')  # Use absolute path for clarity

# Step 3: Install the required packages
!pip install -r requirements.txt

# Import necessary modules
from datasets import load_dataset

# Load and preprocess dataset
dataset = load_dataset("cfilt/iitb-english-hindi")
train_data = dataset['train']

# Properly format list comprehensions
train_sentences_en = [item['translation']['en'] for item in train_data]
train_sentences_hi = [item['translation']['hi'] for item in train_data]

# Separate print statements
print("English Sentences:", train_sentences_en[:5])
print("\nHindi Sentences:", train_sentences_hi[:5])

# Create directories with proper paths
os.makedirs('/content/Models/Mytransformer/weights', exist_ok=True)
os.makedirs('/content/Models/Mytransformer/vocab', exist_ok=True)

# Import config after directory changes
from config import get_config

# Configure settings properly
cfg = get_config()
cfg.update({
    'model_folder': '/content/Models/Mytransformer/weights',
    'tokenizer_file': '/content/Models/Mytransformer/vocab/tokenizer_{0}.json',
    'batch_size': 24,
    'num_epochs': 100,
    'preload': None,

})

# Import and run training
from train import train_model
train_model(cfg)

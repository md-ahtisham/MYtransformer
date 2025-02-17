%capture
!pip install datasets
!pip install tokenizers
!pip install torchmetrics
     
# Step 1: Clone the GitHub repository
!git clone https://github.com/hkproj/pytorch-transformer.git

# Step 2: Change directory to the cloned repository
%cd pytorch-transformer

# Step 3: Install the required packages
!pip install -r requirements.txt

# Load the "cfilt/iitb-english-hindi" dataset
dataset = load_dataset("cfilt/iitb-english-hindi")
     
# Preprocess the dataset as needed
train_data = dataset['train']

# Extract English and Hindi sentences
train_sentences_en = [item['en'] for item in train_data['translation']]
train_sentences_hi = [item['hi'] for item in train_data['translation']]

# Check the first few sentences to verify
print("English Sentences:", train_sentences_en[:5])
print("Hindi Sentences:", train_sentences_hi[:5])
     

# Create directories for model weights and vocab
import os

os.makedirs('/kaggle/working/Models/pytorch-transformer/weights', exist_ok=True)
os.makedirs('/kaggle/working/Models/pytorch-transformer/vocab', exist_ok=True)
     

from config import get_config
cfg = get_config()
cfg['model_folder'] = '..//drive/MyDrive/Models/pytorch-transformer/weights'
cfg['tokenizer_file'] = '..//drive/MyDrive/Models/pytorch-transformer/vocab/tokenizer_{0}.json'
cfg['batch_size'] = 24
cfg['num_epochs'] = 100
cfg['preload'] = None

from train import train_model

train_model(cfg)

# MalUDA

Official implementation of **MalUDA: Unsupervised Domain Adaptation for Robust Malware Detection under Unseen Attacks in IoT Environment**.

---

## Python Environment

- Python 3.11
- Recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate # Linux / macOS
venv\Scripts\activate # Windows
pip install -r requirements.txt


---

## Usage

Run the main training script:

python main.py --source A_Source --target E_Source


**Arguments:**

- `--source` : Name or path of the source dataset  
- `--target` : Name or path of the target dataset  

Optional arguments (examples):

- `--epochs` : Number of training epochs (default: 10)  
- `--lr` : Learning rate (default: 0.001)  

---

## Folder Descriptions

- **utils/** : Helper functions and utilities  
- **data/** : Dataset storage  
- **models/** : Source Extractor, Classifier
- **main.py** : Main script for training and evaluation  
# NetUDA

Official implementation of **NetUDA: A Novel Intrusion Detection Framework via
Unsupervised Domain Adaptation in IoT Network
Environments**.

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
NetUDA included a preprocessing function, so you can just simply put the raw tabular data.
Run the main training script:

python main.py --source A_Source --target E_Source


**Arguments:**

- `--source` : Name or path of the source dataset  
- `--target` : Name or path of the target dataset  

Optional arguments (examples):

- `--epochs` : Number of training epochs (default: 50)  

---

## Folder Descriptions

- **utils/** : Helper functions and utilities  
- **data/** : Dataset storage  
- **models/** : Source Extractor, Classifier
- **main.py** : Main script for training and evaluation  

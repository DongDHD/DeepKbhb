
---

# DeepKbhb: Context-Aware Prediction of Human Lysine β-Hydroxybutyrylation Sites

--- 


## 🚀 Web server
We provide a web server to the users,  which you can access from the [Web server](https://awi.cuhk.edu.cn/~DeepKbhb/).


## 📂 Project Structure

```
DeepKbhb/
├── data/
│   ├── train.csv         # Training dataset
│   └── test.csv
│   └── test_homolog.csv         
├── code/
│   └── best_model.pth    # Pretrained model weights
│   └── best_model_homolog.pth    # Pretrained model weights
│   └── descriptors.py        # Descriptor Feature Extraction
│   └── model.py              # Model architecture
│   └── train.py              # Script to train the model
```

---


## 🛠️ Setup and Installation

1. **Clone the repository**

```bash
git clone https://github.com/DongDHD/DeepKbhb.git
cd DeepKbhb
```
---

2. **Model Training**

Train the model from scratch:

```bash
python train.py
```

# config.py
input_dim = 6
seq_len = 200
num_classes = 2
d_model = 128
nhead = 4
num_layers = 2
dropout = 0.3
batch_size = 64
num_epochs = 125
lr = 1e-4
mission = 'turn2'
mat_path = fr"C:\Users\swl\Desktop\实验数据\onlyIMU\tensor\{mission}_tensor\模式1\{seq_len}\train\combined_tensor_data.mat"
device = "cuda" if __import__('torch').cuda.is_available() else "cpu"

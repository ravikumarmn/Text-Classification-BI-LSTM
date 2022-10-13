runtime_name = "base_bidi_lstm_seq_hidden_mean_loss_mean"
device = 'cuda'

base_dir = "/content/drive/MyDrive/binary_classification/text_classification/"
file_name = "dataset/train.tsv"

vocab_file_name = "dataset/vocab.json"
save_checkpoint_dir = base_dir + "trained_models/"
train_test_data = "dataset/train_test_vocabed.pkl"
word2vec_file = "dataset/word2vectors_32.model" #  "dataset/word2vectors_16.model"
emb_vec_file = "dataset/emb_vec.pkl" 

weight_decay=1e-5
target_columns = [1]
input_column = [3]

max_seq_len = 500
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100

EMBED_SIZE = 32
HIDDEN_SIZE = 32
OUT_DIM = 16

n_labels = 1
patience = 3


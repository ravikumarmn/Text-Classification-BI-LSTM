runtime_name = "base_bidi_lstm_seq_hidden_mean_loss_sum_added_layer_dropout_0.5_relu_imdb_prep_w2v_pre_trained_embedding,added_metrics"
device = 'cpu'

base_dir = "/content/drive/MyDrive/DL_projects/text_classification/"
file_name = "dataset/imdb_dataset.csv"

mapping = {"negative":0,"positive" : 1}
vocab_file_name = "dataset/vocab.json"
save_checkpoint_dir = base_dir + "trained_models/"
train_test_data = "dataset/imdb_train_test_vocabed.pkl"#"dataset/train_test_vocabed.pkl"
word2vec_file = "dataset/prep_word2vectors_32.model"#"dataset/word2vectors_32.model" #  .wordvectors word2vec.wordvectors" dataset/word2vec.wordvectors.vectors.npy
emb_vec_file = "dataset/prep_emb_vec.pkl"#"dataset/emb_vec.pkl" 

weight_decay=1e-5
target_columns = ["sentiment"]
input_column = ["review"]

max_seq_len = 500
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10

EMBED_SIZE = 32
HIDDEN_SIZE = 32
OUT_DIM = 16

n_labels = 1
patience = 3


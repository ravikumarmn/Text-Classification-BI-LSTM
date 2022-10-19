import pandas as pd
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
sw = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer  = PorterStemmer()

def read_csv(base_dir,file_name,input_column,target_columns):
  dataset = pd.read_csv(base_dir+file_name)
  return dataset

def preprocess_text(text):
    text = re.sub(r"[^A-Za-z]"," ",text).strip()
    # text = " ".join([x for x in text.split() if x not in sw])
    # text = " ".join([wordnet_lemmatizer.lemmatize(x) for x in text.split() ])
    # # text = " ".join([porter_stemmer.stem(x) for x in text.split()])
    text = re.sub(r"[^\w\s]", '', text).strip()
    text = text.lower()
    return text

def data_length(dataframe):
    all_words = set()
    all_sentence = list()
    for sentence in dataframe["prep_data"]:
        all_sentence.append(sentence.lower().split())
        for word in sentence.lower().split():
            all_words.add(word)
    len_all_words = len(all_words)
    len_all_sentence = len(all_sentence)
    print(f"Total number of words : {len_all_words}")
    print(f"Total number of sentence : {len_all_sentence}")
    print()
    return all_words,all_sentence

def build_word2vec(all_sentence,embedding_size):
    w2v_model = gensim.models.Word2Vec(sentences=all_sentence,min_count=1,vector_size= embedding_size)
    w2v_model.build_vocab(all_sentence)
    print("Length of samples : ",w2v_model.corpus_count)
    print("Length of vocab   : ",len(w2v_model.wv.key_to_index))
    print("Training and saving model...")
    w2v_model.train(all_sentence,total_examples=w2v_model.corpus_count,epochs=w2v_model.epochs)
    w2v_model.save(f"{config.local_base_dir}dataset/prep_word2vectors_{config.EMBED_SIZE}.model")
    wordvecs = KeyedVectors.load(f"{config.local_base_dir}dataset/prep_word2vectors_{config.EMBED_SIZE}.model")    
    all_words = list(wordvecs.wv.key_to_index.keys())
    # wordvecs.wv.index_to_key = {v:k for k,v in wordvecs.wv.key_to_index.items()}
    word2index = {k:v+1 for k,v in wordvecs.wv.key_to_index.items()}
    index2word = {v:k for k,v in word2index.items()}
    
    matrix_vec = np.zeros((len(word2index)+1,config.EMBED_SIZE))
    for word,idx in word2index.items():
        vector_x = wordvecs.wv[word]
        matrix_vec[idx,:] = vector_x    
    pickle_data = {
    "word2index" : word2index,
    "index2word" : index2word,
    "embedding_vector" : matrix_vec
    }
    pickle.dump(pickle_data,open(config.local_base_dir+"dataset/prep_emb_vec.pkl",'wb'))
    print("Done")
    print(f'Model saved to {config.local_base_dir+"dataset/prep_emb_vec.pkl"}')
    

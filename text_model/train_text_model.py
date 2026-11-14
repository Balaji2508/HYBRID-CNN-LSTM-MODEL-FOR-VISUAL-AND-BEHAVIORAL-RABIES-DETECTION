import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle
import os

# Config
DATA_FILE = r"D:\balaji v2\Rabbies_Detection_Project\data\symptoms.csv"
MODEL_SAVE_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\text_model.h5"
TOKENIZER_SAVE_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\tokenizer.pickle"
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
EPOCHS = 10
BATCH_SIZE = 32

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        return

    # Check output dir
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Check for empty data
    if df.empty:
        print("Dataset is empty!")
        return

    # Tokenize
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['text'].astype(str).values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    # Save tokenizer
    with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    X = tokenizer.texts_to_sequences(df['text'].astype(str).values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    
    Y = pd.get_dummies(df['label']).values
    print('Shape of label tensor:', Y.shape)
    
    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Build Model
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    # Train
    history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    
    # Save
    model.save(MODEL_SAVE_PATH)
    print("Text model saved to", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()

import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_model_and_tokenizer(model_path, tokenizer_path):
    """
    Load the trained model and tokenizer from the specified paths.
    
    Parameters:
        model_path (str): Path to the trained model file.
        tokenizer_path (str): Path to the tokenizer pickle file.
    
    Returns:
        model (tf.keras.Model): Loaded model.
        tokenizer (Tokenizer): Loaded tokenizer.
    """
    try:
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as file:
            tokenizer = pickle.load(file)
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def prepare_input_sequence(tokenizer, text, max_length=3):
    """
    Prepares the input text sequence by tokenizing and padding it.
    
    Parameters:
        tokenizer (Tokenizer): Tokenizer to convert text to sequences.
        text (str): Input text.
        max_length (int): Maximum length of the input sequence.
    
    Returns:
        np.ndarray: Padded sequence ready for model input.
    """
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=max_length)

def predict_next_word(model, tokenizer, input_text, max_length=3):
    """
    Predict the next word for a given input text sequence.
    
    Parameters:
        model (tf.keras.Model): Loaded prediction model.
        tokenizer (Tokenizer): Loaded tokenizer.
        input_text (str): Input text sequence.
        max_length (int): Maximum sequence length for padding.
    
    Returns:
        str: Predicted next word.
    """
    sequence = prepare_input_sequence(tokenizer, input_text, max_length)
    predictions = model.predict(sequence)
    predicted_index = np.argmax(predictions, axis=-1)[0]
    
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
    predicted_word = reverse_word_index.get(predicted_index, "<unknown>")
    
    print(f"Predicted word: {predicted_word}")
    return predicted_word

def main():
    model_path = "C:/Users/Lenovo/New folder (6)/notebooks/next_words.keras"
    tokenizer_path = "C:/Users/Lenovo/New folder (6)/notebooks/token1.pkl"
    
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return
    
    print("Enter text to predict the next word (or type 'exit' to quit):")
    while True:
        user_input = input("Input text: ")
        if user_input.lower() == "exit":
            print("Exiting.")
            break
        
        # Extract last three words for prediction
        input_text = " ".join(user_input.split()[-3:])
        predict_next_word(model, tokenizer, input_text)

if __name__ == "__main__":
    main()

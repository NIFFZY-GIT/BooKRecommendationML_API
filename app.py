import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process # Make sure this import is here

# --- 1. DEFINE MODEL ARCHITECTURE ---
def create_recommender_model(num_users, num_books, embedding_size=50):
    user_input = tf.keras.layers.Input(shape=[1], name='UserInput')
    user_embedding = tf.keras.layers.Embedding(input_dim=num_users, output_dim=embedding_size, name='UserEmbedding')(user_input)
    user_vector = tf.keras.layers.Flatten(name='FlattenUserVec')(user_embedding)

    book_input = tf.keras.layers.Input(shape=[1], name='BookInput')
    book_embedding = tf.keras.layers.Embedding(input_dim=num_books, output_dim=embedding_size, name='BookEmbedding')(book_input)
    book_vector = tf.keras.layers.Flatten(name='FlattenBookVec')(book_embedding)

    dot_product = tf.keras.layers.Dot(axes=1, name='DotProduct')([user_vector, book_vector])
    
    model = tf.keras.models.Model(inputs=[user_input, book_input], outputs=dot_product)
    return model

# --- 2. SETUP: Load data, create model, and load weights ---
print("Flask app starting... attempting to load data and model weights.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_ENCODER_PATH = os.path.join(BASE_DIR, 'user_encoder.pkl')
BOOK_ENCODER_PATH = os.path.join(BASE_DIR, 'book_encoder.pkl')
BOOKS_CSV_PATH = os.path.join(BASE_DIR, 'books.csv')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'book_recommender.weights.h5')

try:
    with open(USER_ENCODER_PATH, 'rb') as f:
        user_encoder = pickle.load(f)
    
    with open(BOOK_ENCODER_PATH, 'rb') as f:
        book_encoder = pickle.load(f)

    books_df = pd.read_csv(BOOKS_CSV_PATH, sep=';', on_bad_lines='skip', encoding='latin-1')
    books_df.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']

    num_users = len(user_encoder.classes_)
    num_books = len(book_encoder.classes_)

    model = create_recommender_model(num_users, num_books)
    model.load_weights(WEIGHTS_PATH)

    book_embedding_weights = model.get_layer('BookEmbedding').get_weights()[0]
    print("✅ Model, weights, and data loaded successfully! The server is ready.")

except Exception as e:
    print(f"❌ An error occurred during startup: {e}")
    raise e


# --- 3. The Internal Recommendation Function ---
def get_recommendations_for_title(book_title, num_recs=5):
    """
    This is the internal logic that gets recommendations for a PERFECTLY MATCHED title.
    """
    try:
        book_id_encoded = book_encoder.transform([book_title])[0]
        book_vec = book_embedding_weights[book_id_encoded].reshape(1, -1)
    except (ValueError, IndexError):
        return {"error": f"Book '{book_title}' not found or invalid."}

    similarities = cosine_similarity(book_vec, book_embedding_weights)[0]
    similar_book_indices = np.argsort(similarities)[::-1]
    recommendations = []
    rec_count = 0
    for idx in similar_book_indices:
        if idx == book_id_encoded:
            continue
        recommended_title = book_encoder.inverse_transform([idx])[0]
        similarity_score = float(similarities[idx])
        book_details = books_df[books_df['Book-Title'] == recommended_title].drop_duplicates('Book-Title')
        if not book_details.empty:
            rec_data = {
                "title": recommended_title,
                "author": book_details['Book-Author'].values[0],
                "year": str(book_details['Year-Of-Publication'].values[0]),
                "publisher": book_details['Publisher'].values[0],
                "image_url": book_details['Image-URL-M'].values[0],
                "similarity_score": round(similarity_score, 4)
            }
            recommendations.append(rec_data)
        rec_count += 1
        if rec_count >= num_recs:
            break
    return {"recommendations": recommendations}


# --- 4. The Flask API Endpoints ---
app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Book Recommendation API</h1><p>Use the /recommend endpoint. Example: /recommend?title=The+Da+Vinci+Code</p>"

# THIS IS THE ONLY /recommend ROUTE
@app.route('/recommend', methods=['GET'])
def recommend():
    """
    The main public endpoint that uses fuzzy matching to find the best book
    and then gets recommendations for it.
    """
    user_query = request.args.get('title')
    
    if not user_query:
        return jsonify({"error": "Please provide a 'title' query parameter."}), 400

    # --- FUZZY MATCHING LOGIC ---
    all_known_titles = book_encoder.classes_
    best_match, score = process.extractOne(user_query, all_known_titles)

    if score < 70:
        return jsonify({"error": f"Could not find a close match for '{user_query}'. Please try a different title."})

    # Use the internal function with the BEST MATCH to get recommendations.
    results = get_recommendations_for_title(best_match)
    
    # Add the matched title to the response so the user knows what they got recommendations for.
    results['matched_title'] = best_match
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
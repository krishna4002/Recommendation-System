# scripts/load_user_history.py
import pandas as pd
def get_user_likes(user_id, rating_path="data/ratings.csv", book_path="data/books.csv", min_rating=4):
    ratings = pd.read_csv(rating_path)
    books = pd.read_csv(book_path)
    user_rated = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= min_rating)]
    return books[books['book_id'].isin(user_rated['book_id'])][['title', 'authors', 'description']]
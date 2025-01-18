import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime

def load_data():
    # Baca dataset
    movies_df = pd.read_csv('./data/movies.csv')
    ratings_df = pd.read_csv('./data/ratings.csv')
    
    # Tampilkan informasi dasar
    print("\nInformasi Dataset Film:")
    print(movies_df.info())
    print("\nSample data film:")
    print(movies_df.head())
    
    print("\nInformasi Dataset Rating:")
    print(ratings_df.info())
    print("\nSample data rating:")
    print(ratings_df.head())
    
    return movies_df, ratings_df

if __name__ == "__main__":
    movies, ratings = load_data()

# fase 2

def create_genre_matrix(movies_df):
    # Mengubah string genre menjadi kolom-kolom
    genres = movies_df['genres'].str.split('|', expand=True).stack()
    # Mendapatkan list unik genre
    unique_genres = genres.unique()
    
    # Membuat matrix genre (1 untuk genre yang dimiliki, 0 untuk yang tidak)
    genre_matrix = np.zeros((len(movies_df), len(unique_genres)))
    
    for i, movie_genres in enumerate(movies_df['genres'].str.split('|')):
        for genre in movie_genres:
            genre_idx = np.where(unique_genres == genre)[0][0]
            genre_matrix[i, genre_idx] = 1
            
    return pd.DataFrame(genre_matrix, columns=unique_genres)

def get_movie_recommendations(movie_title, movies_df, n_recommendations=5):
    # Dapatkan genre matrix
    genre_matrix = create_genre_matrix(movies_df)
    
    # Hitung similarity antar film
    similarity_matrix = cosine_similarity(genre_matrix)
    
    # Dapatkan index film yang dicari
    movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
    
    # Hitung similarity score untuk semua film
    movie_similarities = list(enumerate(similarity_matrix[movie_idx]))
    
    # Urutkan film berdasarkan similarity
    similar_movies = sorted(movie_similarities, key=lambda x: x[1], reverse=True)
    
    # Ambil n rekomendasi teratas (skip film pertama karena itu film yang sama)
    recommendations = []
    for i in range(1, n_recommendations + 1):
        movie_idx = similar_movies[i][0]
        recommendations.append({
            'title': movies_df.iloc[movie_idx]['title'],
            'genres': movies_df.iloc[movie_idx]['genres'],
            'similarity_score': similar_movies[i][1]
        })
    
    return recommendations

# Di bagian main, tambahkan:
#if __name__ == "__main__":
#    movies, ratings = load_data()
#    
#    # Contoh penggunaan
#    movie_title = "Jumanji (1995)"
#    recommendations = get_movie_recommendations(movie_title, movies)
#    
#    print(f"\nRekomendasi untuk film {movie_title}:")
#    for i, rec in enumerate(recommendations, 1):
#        print(f"{i}. {rec['title']}")
#        print(f"   Genre: {rec['genres']}")
#        print(f"   Similarity Score: {rec['similarity_score']:.2f}\n")


#fase 3

def extract_year(title):
    """Ekstrak tahun dari judul film"""
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else 0

def create_feature_matrix(movies_df, ratings_df):
    # 1. Genre Matrix
    genres = movies_df['genres'].str.split('|', expand=True).stack()
    unique_genres = genres.unique()
    genre_matrix = np.zeros((len(movies_df), len(unique_genres)))
    
    for i, movie_genres in enumerate(movies_df['genres'].str.split('|')):
        for genre in movie_genres:
            genre_idx = np.where(unique_genres == genre)[0][0]
            genre_matrix[i, genre_idx] = 1
    
    # 2. Rating Features
    avg_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movies_with_ratings = pd.merge(movies_df, avg_ratings, on='movieId', how='left')
    movies_with_ratings[['mean', 'count']] = movies_with_ratings[['mean', 'count']].fillna(0)
    
    # 3. Year Feature
    movies_with_ratings['year'] = movies_with_ratings['title'].apply(extract_year)
    current_year = datetime.now().year
    movies_with_ratings['year_factor'] = movies_with_ratings['year'].apply(
        lambda x: 1 / (1 + 0.1 * (current_year - x)) if x != 0 else 0
    )
    
    # Membuat matrix fitur gabungan
    genre_df = pd.DataFrame(genre_matrix, columns=unique_genres)
    
    # Normalisasi fitur
    count_max = movies_with_ratings['count'].max()
    movies_with_ratings['count_normalized'] = movies_with_ratings['count'] / count_max
    
    # Gabungkan semua fitur dengan pembobotan
    feature_matrix = pd.concat([
        genre_df * 0.4,  # Genre weight
        movies_with_ratings[['mean']].mul(0.3),  # Rating weight
        movies_with_ratings[['count_normalized']].mul(0.2),  # Popularity weight
        movies_with_ratings[['year_factor']].mul(0.1)  # Year weight
    ], axis=1)
    
    return feature_matrix, movies_with_ratings

def get_enhanced_recommendations(movie_title, movies_df, ratings_df, n_recommendations=5):
    # Buat feature matrix
    feature_matrix, movies_with_ratings = create_feature_matrix(movies_df, ratings_df)
    
    # Hitung similarity
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # Dapatkan index film yang dicari
    movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
    
    # Hitung similarity scores
    movie_similarities = list(enumerate(similarity_matrix[movie_idx]))
    
    # Urutkan film berdasarkan similarity
    similar_movies = sorted(movie_similarities, key=lambda x: x[1], reverse=True)
    
    # Siapkan rekomendasi
    recommendations = []
    for i in range(1, n_recommendations + 1):
        movie_idx = similar_movies[i][0]
        movie_data = movies_with_ratings.iloc[movie_idx]
        recommendations.append({
            'title': movie_data['title'],
            'genres': movie_data['genres'],
            'similarity_score': similar_movies[i][1],
            'avg_rating': movie_data['mean'],
            'num_ratings': movie_data['count'],
            'year': movie_data['year']
        })
    
    return recommendations

if __name__ == "__main__":
    movies, ratings = load_data()
    
    # Contoh penggunaan
    movie_title = "Toy Story (1995)"
    recommendations = get_enhanced_recommendations(movie_title, movies, ratings)
    
    print(f"\nRekomendasi yang Ditingkatkan untuk film {movie_title}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Genre: {rec['genres']}")
        print(f"   Rating Rata-rata: {rec['avg_rating']:.2f} dari {int(rec['num_ratings'])} ulasan")
        print(f"   Tahun: {rec['year']}")
        print(f"   Similarity Score: {rec['similarity_score']:.2f}")
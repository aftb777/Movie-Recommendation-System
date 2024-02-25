import numpy as np 
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data collection and processing
#loading the data fromt the CSV file to pandas dataframe

movies_data = pd.read_csv('/Users/aftaabmulla/python projects/movies.csv')

#printing the first 5 rows of the dataframe

movies_data.head()

#no of rows and columns in the dataframe

movies_data.shape

#selecting the relevant features for recommendations

selected_features = ['genres', 'keyword', 'tagline', 'cast', 'director']
print(selected_features)

#replacing the null values with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
    
# Comining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keyword']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

# Converting the text data to feature vectors

vectorizer = TfidfVectorizer()

feature_vector = vectorizer.fit_transform(combined_features)

print(feature_vector)

# Cosine similarity - getting similarity scored using cosine similarity

similarity = cosine_similarity(feature_vector)
print(similarity)
print(similarity.shape)

# Getting movie name 

movies_name = input("Enter your favourite movie name : ")

# Creating a list with all the movie names given in the dataset

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)

# Finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movies_name, list_of_all_titles)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

# Find the index of the movie with title

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)

# getting list of similar movies 

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

len(similarity_score)

# Sorting the movies based on thier similarity score

sorted_similar_movies = sorted(similarity_score,key = lambda x:x[1], reverse= True)
print(sorted_similar_movies)

# Print the name of similar movies based on the index

print('Movies Suggested for you : \n')
i =1 
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if(i<30):
        print(i, '.',title_from_index)
        i += 1
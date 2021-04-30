import pandas as pd
import numpy as ny
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv('movie_dataset.csv', engine='python')
#print (data.columns) 


features=['keywords','cast','genres','director']

for four in features:
	data[four]=data[four ].fillna('')

def combined_features(row):
	try:	 
		return row['keywords']+' '+row['cast']+' '+row['genres']+' '+row['director']
	except:
		print('error:',row)

data['combine_features']=data.apply(combined_features,axis=1)
#rint('combined features:',data['combine_features'].head()) 

c=CountVectorizer()
matrix=c.fit_transform(data['combine_features']) 
similarity=cosine_similarity(matrix)

 
def get_title_from_index(index):
 	return data[data.index==index]['title'].values[0]

def get_index_from_title(title):
 	return data[data.title==title]['index'].values[0]

user_likes='Ted'


movie_index=get_index_from_title(user_likes)

similar_movies=list(enumerate(similarity[movie_index]))	

sorted_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)
i=0
for movies in sorted_movies:
	print(get_title_from_index(movies[0]))
	i+=1
	if i>30:
		break 
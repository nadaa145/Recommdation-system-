from fastapi import FastAPI 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_text
from recommender import content_rec_df
from classifier import random_forest_classifier
import os

BASE_DIR = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'udemy_courses.csv'))

app = FastAPI()
data['course_title'] = data['course_title'].apply(clean_text)
cv = CountVectorizer(max_features=3000)
vects = cv.fit_transform(data['course_title'])
sim = cosine_similarity(vects)

@app.get('/')
def root():
    return {"API is running successfully"}

@app.get('/recommend')
def recommend_courses(course: str, top_n: int = 10):
    course = course.strip() 
    try:
        recs_df = content_rec_df(course=course, data=data, top_n=top_n)
        if recs_df is None or recs_df.empty:
            return {"error": "Course not found"}
        final_df = random_forest_classifier(recs_df)
        return final_df
    except Exception as e:
        return {"error": str(e)}

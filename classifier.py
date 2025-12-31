import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

def create_labels(row):
    if row['similarity_score'] >= 0.9:
        return 2
    if row['similarity_score'] >= 0.3:
        return 1
    return 0
    
def random_forest_classifier(recs_df):
    recs_df['relevance'] = recs_df.apply(create_labels, axis=1)
    cv = CountVectorizer(max_features=3000)
    text_vectors = cv.fit_transform(recs_df['course_title'])
    numerical_features = recs_df[['similarity_score', 'num_subscribers', 'num_reviews', 'pop_score']].values
    X = hstack([text_vectors, numerical_features])
    y = recs_df['relevance']
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    predictions = rf.predict(X)
    recs_df['predicted_suitability'] = predictions
    label_map = {2: "High", 1: "Medium", 0: "Low"}
    recs_df['predicted_suitability'] = recs_df['predicted_suitability'].map(label_map)
    recs_df

    return recs_df.to_dict(orient='records')

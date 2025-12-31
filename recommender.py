import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_rec_df(course, data, top_n=10):
    processed_course = nfx.remove_stopwords(course)
    processed_course = nfx.remove_special_characters(processed_course)

    if processed_course not in data['course_title'].values:
        return pd.DataFrame()
    data['pop_score'] = (
        0.6*data['num_subscribers']/data['num_subscribers'].max() +
        0.4*data['num_reviews']/data['num_reviews'].max())
    cv = CountVectorizer(max_features=3000)
    vectors = cv.fit_transform(data['course_title'])
    similarity_matrix = cosine_similarity(vectors)
    course_index = data[data['course_title'] == processed_course].index[0]
    distances = similarity_matrix[course_index]
    combined_score = [
        (i, distances[i]*0.7 + data.iloc[i].pop_score*0.3)
        for i in range(len(distances)) if i != course_index]
    top_courses = sorted(combined_score, key=lambda x: x[1], reverse=True)[:top_n]
    
    recs_df = pd.DataFrame({
        'course_title': [data.iloc[i[0]].course_title for i in top_courses],
        'similarity_score': [distances[i[0]] for i in top_courses],
        'num_subscribers': [data.iloc[i[0]].num_subscribers for i in top_courses],
        'num_reviews': [data.iloc[i[0]].num_reviews for i in top_courses],
        'pop_score': [data.iloc[i[0]].pop_score for i in top_courses]
    })

    return recs_df

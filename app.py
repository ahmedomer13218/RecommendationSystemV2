import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import pickle

# for AutoEncoder 
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)  # Ensure y_true is float32
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    diff = y_pred - y_true
    sqdiff = diff * diff * mask
    sse = tf.reduce_sum(sqdiff)
    n = tf.reduce_sum(mask)
    return sse / n

# Load models and data
def load_models():
    with open('item_item-collaberative_filtering/usermovie2rating.pkl', 'rb') as f:
        loaded_usermovie2rating_m1 = pickle.load(f)

    with open('item_item-collaberative_filtering/usermovie2rating_test.pkl', 'rb') as f:
        loaded_usermovie2rating_test_m1 = pickle.load(f)

    with open('item_item-collaberative_filtering/neighbors.pkl', 'rb') as f:
        loaded_neighbors_m1 = pickle.load(f)

    with open('item_item-collaberative_filtering/averages.pkl', 'rb') as f:
        loaded_averages_m1 = pickle.load(f)

    with open('item_item-collaberative_filtering/deviations.pkl', 'rb') as f:
        loaded_deviations_m1 = pickle.load(f)

    with open('user_user_collaberative_filtring/usermovie2rating.pkl', 'rb') as f:
        loaded_usermovie2rating_m2 = pickle.load(f)

    with open('user_user_collaberative_filtring/usermovie2rating_test.pkl', 'rb') as f:
        loaded_usermovie2rating_test_m2 = pickle.load(f)

    with open('user_user_collaberative_filtring/neighbors.pkl', 'rb') as f:
        loaded_neighbors_m2 = pickle.load(f)

    with open('user_user_collaberative_filtring/averages.pkl', 'rb') as f:
        loaded_averages_m2 = pickle.load(f)

    with open('user_user_collaberative_filtring/deviations.pkl', 'rb') as f:
        loaded_deviations_m2 = pickle.load(f)

    with open('matrix_factorization/mu.pkl', 'rb') as f: #    mu 10 , df 11 , 3 12 , 4 13, 5 14
        mu = pickle.load(f)

    df=pd.read_csv('data/edited_rating.csv')

    model_3 = load_model('matrix_factorization/MF_model.h5', custom_objects={'mse': MeanSquaredError()})
    model_4 = load_model('matrix_factorization/MF_model_ANN.h5', custom_objects={'mse': MeanSquaredError()})
    model_5 = load_model('matrix_factorization/MF_model_lin_res_ANN.h5', custom_objects={'mse': MeanSquaredError()})

    model_6 = load_model('matrix_factorization/model_AutoEncoder.h5', custom_objects={'custom_loss': custom_loss})

    A = load_npz("matrix_factorization/Atrain.npz")

    return (loaded_usermovie2rating_m1, loaded_usermovie2rating_test_m1, 
            loaded_neighbors_m1, loaded_averages_m1, loaded_deviations_m1,loaded_usermovie2rating_m2, loaded_usermovie2rating_test_m2, 
            loaded_neighbors_m2, loaded_averages_m2, loaded_deviations_m2,mu,df,model_3,model_4,model_5,model_6,A)




###########################  Model Prediction Functions ########################

##### Prediction function for Item_Item Based

def predict_model1(i, u, usermovie2rating_train_m1, usermovie2rating_test_m1, loaded_neighbors_m1, loaded_averages_m1, loaded_deviations_m1):
    numerator = 0
    denominator = 0
    for neg_w, j in loaded_neighbors_m1[i]:
        try:
            numerator += -neg_w * loaded_deviations_m1[j][u]
            denominator += abs(neg_w)
        except KeyError:
            pass

    if denominator == 0:
        prediction = loaded_averages_m1[i]
    else:
        prediction = numerator / denominator + loaded_averages_m1[i]

    prediction = min(5, prediction)
    prediction = max(0.5, prediction)

    if (i, u) in usermovie2rating_train_m1 or (i, u) in usermovie2rating_test_m1:
        actual_value = usermovie2rating_train_m1.get((i, u), usermovie2rating_test_m1.get((i, u)))
        return f"user with ID {u} watched the movie with ID {i} and rated it : {actual_value} and the prediction value is: {prediction:.3f}"
    else:
        return f"user with ID {u} have NOT watched the movie with ID {i} , and the prediction value is: {prediction:.3f}"



##### Prediction function for User_User Based

def predict_model2(i, m, usermovie2rating_train_m2, usermovie2rating_test_m2, loaded_neighbors_m2, loaded_averages_m2, loaded_deviations_m2):
    # Calculate the weighted sum of deviations
    numerator = 0
    denominator = 0
    for neg_w, j in loaded_neighbors_m2[i]:
        # Remember, the weight is stored as its negative
        try:
            numerator += -neg_w * loaded_deviations_m2[j][m]
            denominator += abs(neg_w)
        except KeyError:
            # Neighbor may not have rated the same movie
            pass

    if denominator == 0:
        prediction = loaded_averages_m2[i]
    else:
        prediction = numerator / denominator + loaded_averages_m2[i]
    
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)  # Min rating is 0.5
    
    if (i, m) in usermovie2rating_train_m2 or (i, m) in usermovie2rating_test_m2:
        actual_value = usermovie2rating_train_m2.get((i, m), usermovie2rating_test_m2.get((i, m)))
        return f"user with ID {i} watched the movie with ID {m} and rated it : {actual_value} and the prediction value is: {prediction:.3f}"
    else:
        return f"user with ID {i} have NOT watched the movie with ID {m} , and the prediction value is: {prediction:.3f}"


##### Prediction function for MF & ANN & ANN+Res

def predict_model_3_4_5(model, user_id, movie_id, mu,df):

    # Prepare inputs for the model (as arrays)
    user_input = np.array([user_id])  # Shape should be (1,)
    movie_input = np.array([movie_id])  # Shape should be (1,)

    # Predict the rating deviation from the mean
    predicted_rating = model.predict([user_input, movie_input])

    # Add the global mean rating back to the prediction
    final_predicted_rating = predicted_rating[0][0] + mu

    actual_value=df[(df.User == user_id) & (df.movie_idx == movie_id)].Rating.values
    
    if len(actual_value)!=0:
        return f"user with ID {user_id} watched the movie with ID {movie_id} and rated it : {actual_value} and the prediction value is: {final_predicted_rating:.3f}"
    else:
        return f"user with ID {user_id} have NOT watched the movie with ID {movie_id} , and the prediction value is: {final_predicted_rating:.3f}"


##### Prediction function for AutoEncoder
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)  # Ensure y_true is float32
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    diff = y_pred - y_true
    sqdiff = diff * diff * mask
    sse = tf.reduce_sum(sqdiff)
    n = tf.reduce_sum(mask)
    return sse / n

def predict_model_6(model,user_id, movie_id, mu , A,df):
    # Get the user ratings (row for user_id)
    user_ratings = A[user_id].toarray()  # Convert sparse matrix to dense if necessary
    
    # Center the data (subtract the global average, mu)
    user_ratings = user_ratings - mu * (user_ratings > 0)
    
    # Get the prediction for this user (entire row)
    user_prediction = model.predict(user_ratings)

    prediction_rescaled = user_prediction + mu
    
    # Retrieve the predicted rating for the specific movie_id
    predicted_rating = prediction_rescaled[0][movie_id]
    
    actual_value=df[(df.User == user_id) & (df.movie_idx == movie_id)].Rating.values
    
    if len(actual_value)!=0:
        return f"user with ID {user_id} watched the movie with ID {movie_id} and rated it : {actual_value} and the prediction value is: {predicted_rating:.3f}"
    else:
        return f"user with ID {user_id} have NOT watched the movie with ID {movie_id} , and the prediction value is: {predicted_rating:.3f}"





###########################  Recommendation  Functions ###########################


##### Recommendation function for Item_Item Based

def recommend_movies_model1(user_id, loaded_usermovie2rating, loaded_neighbors, loaded_averages, loaded_deviations, movie_limit):
    predictions = {}
    # Iterate over all movies up to the movie_limit
    for movie_id in range(movie_limit + 1):  # Adjust for 0-indexing
        if (movie_id, user_id) not in loaded_usermovie2rating:
            prediction = predict_model1(movie_id, user_id, 
                                         loaded_usermovie2rating, 
                                         loaded_usermovie2rating, 
                                         loaded_neighbors, 
                                         loaded_averages, 
                                         loaded_deviations)
            # Extract prediction value for sorting
            rating = float(prediction.split("and the prediction value is: ")[-1])
            predictions[movie_id] = rating

    # Sort predictions by rating and get the top 5
    top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_movies


##### Recommendation function for User_User Based

def recommend_movies_model2(user_id, loaded_usermovie2rating, loaded_neighbors, loaded_averages, loaded_deviations, movie_limit):
    predictions = {}
    # Iterate over all movies up to the movie_limit
    for movie_id in range(movie_limit + 1):  # Adjust for 0-indexing
        if (movie_id, user_id) not in loaded_usermovie2rating:
            prediction = predict_model2(user_id, movie_id, 
                                         loaded_usermovie2rating, 
                                         loaded_usermovie2rating, 
                                         loaded_neighbors, 
                                         loaded_averages, 
                                         loaded_deviations)
            # Extract prediction value for sorting
            rating = float(prediction.split("and the prediction value is: ")[-1])
            predictions[movie_id] = rating

    # Sort predictions by rating and get the top 5
    top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_movies


##### Recommendation function for MF & ANN & ANN+Res

def recommend_movies_model_3_4_5(user_id, model, df, mu, movie_limit):
    predictions = {}
    # Iterate over all movies up to the movie_limit
    for movie_id in range(500 + 1):  # Adjust for 0-indexing
        prediction = predict_model_3_4_5(model, user_id, movie_id, mu, df)
        rating = float(prediction.split("and the prediction value is: ")[-1])
        predictions[movie_id] = rating

    # Sort predictions by rating and get the top 5
    top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_movies


##### Recommendation function for AutoEncoder

def recommend_movies_model_6(model,user_id , mu , A):
    # Get the user ratings (row for user_id)
    user_ratings = A[user_id].toarray()  # Convert sparse matrix to dense if necessary
    
    # Center the data (subtract the global average, mu)
    user_ratings = user_ratings - mu * (user_ratings > 0)
    
    # Get the prediction for this user (entire row)
    user_prediction = model.predict(user_ratings)

    prediction_rescaled = user_prediction + mu

    # Convert the predictions to a list of (movie_id, predicted_rating)
    movie_ids = np.arange(len(prediction_rescaled[0]))  # Assuming movies are indexed from 0 to len(user_prediction)-1
    predictions = list(zip(movie_ids, prediction_rescaled[0]))

    # Sort predictions by rating and get the top 10
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
    
    return top_movies






#####  Initialize session state to keep track of which page the user is on

if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Set initial page to 'home'

if 'model' not in st.session_state:
    st.session_state.model = None  # Initialize model in session state

if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = load_models()  # Load models on first run





#####  Home page: Option to choose between 'Predict Rating' or 'Recommend Movies'
def home_page():
    st.title("Movie Recommendation System")
    st.subheader("Choose an option")
    
    option = st.radio("Select an option:", ('Predict Rating', 'Recommend Movies'))
    
    # Save selected option to session state and navigate to model selection page
    if st.button("Next"):
        st.session_state.option = option
        st.session_state.page = 'model_selection'





##### Model selection page using buttons
def model_selection_page():
    st.title("Model Selection")
    st.subheader("Choose a Model")

    # Create buttons for model selection
    col1, col2, col3 = st.columns(3)  # Organize buttons into columns for better layout
    
    with col1:
        if st.button("Item_Item Based"):
            st.session_state.model = 'Item_Item Based'
    
    with col2:
        if st.button("User_User Based"):
            st.session_state.model = 'User_User Based'
    
    with col3:
        if st.button("Matrix_Factorization"):
            st.session_state.model = 'Matrix_Factorization'

    with col1:
        if st.button("MF_ANN"):
            st.session_state.model = 'MF_ANN'
    
    with col2:
        if st.button("MF_ANN+Residual"):
            st.session_state.model = 'MF_ANN+Residual'
    
    with col3:
        if st.button("AutoEncoder"):
            st.session_state.model = 'AutoEncoder'

    # Move to the next page if a model is selected
    if st.session_state.model:
        st.write(f"Selected model: {st.session_state.model}")  # Show the selected model (for debugging)
        if st.button("Next"):
            st.session_state.page = 'input'




####  Set limits based on the selected model (adjusted for 0-indexing)
def set_limits(model):
    if model == 'User_User Based':
        return (999, 799)  
    elif model == 'Item_Item Based':
        return (5999, 1999)  
    else:
        return (6039, 3705)  
    




######   Input page based on user's initial option selection

def input_page():
    st.title("Input Details")
    
    # Get the limits based on the selected model
    user_limit, movie_limit = set_limits(st.session_state.model)
    st.write(f"User limit: {user_limit}, Movie limit: {movie_limit}")  # Debugging print
    
    if st.session_state.option == 'Predict Rating':
        st.subheader("Predict User Rating")
        user_id = st.number_input(f"Enter User ID (0 to {user_limit}):", min_value=0, max_value=user_limit, step=1)
        movie_id = st.number_input(f"Enter Movie ID (0 to {movie_limit}):", min_value=0, max_value=movie_limit, step=1)
        
        if st.button("Predict"):
            if st.session_state.model == 'Item_Item Based':
                output = predict_model1(movie_id, user_id, 
                                         st.session_state.loaded_data[0], 
                                         st.session_state.loaded_data[1], 
                                         st.session_state.loaded_data[2], 
                                         st.session_state.loaded_data[3], 
                                         st.session_state.loaded_data[4])
                
            elif st.session_state.model == 'User_User Based':
                output = predict_model2(user_id,movie_id, 
                                         st.session_state.loaded_data[5], 
                                         st.session_state.loaded_data[6], 
                                         st.session_state.loaded_data[7], 
                                         st.session_state.loaded_data[8], 
                                         st.session_state.loaded_data[9])  
                
            elif st.session_state.model == 'Matrix_Factorization' :
                output = predict_model_3_4_5( st.session_state.loaded_data[12],
                                         user_id,movie_id, 
                                         st.session_state.loaded_data[10], 
                                         st.session_state.loaded_data[11])  
                
            elif st.session_state.model == 'MF_ANN' :
                output = predict_model_3_4_5( st.session_state.loaded_data[13],
                                         user_id,movie_id, 
                                         st.session_state.loaded_data[10], 
                                         st.session_state.loaded_data[11])  
                
            elif st.session_state.model == 'MF_ANN+Residual' :
                output = predict_model_3_4_5( st.session_state.loaded_data[14],
                                         user_id,movie_id, 
                                         st.session_state.loaded_data[10], 
                                         st.session_state.loaded_data[11])  
            else: 
                output = predict_model_6( st.session_state.loaded_data[15],
                                         user_id,movie_id, 
                                         st.session_state.loaded_data[10], 
                                         st.session_state.loaded_data[16], 
                                         st.session_state.loaded_data[11])  

            st.write(output)
    
    elif st.session_state.option == 'Recommend Movies':
        st.subheader("Recommend Movies")
        user_id = st.number_input(f"Enter User ID (0 to {user_limit}):", min_value=0, max_value=user_limit, step=1)
        
        if st.button("Recommend"):
            if st.session_state.model == 'Item_Item Based':
                top_movies = recommend_movies_model1(user_id, 
                                           st.session_state.loaded_data[0], 
                                           st.session_state.loaded_data[2], 
                                           st.session_state.loaded_data[3], 
                                           st.session_state.loaded_data[4],
                                           movie_limit)
            elif st.session_state.model == 'User_User Based':
                top_movies = recommend_movies_model2(user_id, 
                                           st.session_state.loaded_data[5], 
                                           st.session_state.loaded_data[7], 
                                           st.session_state.loaded_data[8], 
                                           st.session_state.loaded_data[9],
                                           movie_limit) 
            
            elif st.session_state.model == 'Matrix_Factorization' :
                top_movies = recommend_movies_model_3_4_5( user_id,
                                             st.session_state.loaded_data[12],
                                             st.session_state.loaded_data[11], 
                                             st.session_state.loaded_data[10],
                                             movie_limit)  
                
            elif st.session_state.model == 'MF_ANN' :
                top_movies = recommend_movies_model_3_4_5(user_id, 
                                             st.session_state.loaded_data[13],
                                             st.session_state.loaded_data[11], 
                                             st.session_state.loaded_data[10],
                                             movie_limit)  
                
            elif st.session_state.model == 'MF_ANN+Residual' :
                top_movies = recommend_movies_model_3_4_5(user_id, 
                                            st.session_state.loaded_data[14],
                                            st.session_state.loaded_data[11], 
                                            st.session_state.loaded_data[10],
                                            movie_limit)  
            else: 
                top_movies = recommend_movies_model_6( st.session_state.loaded_data[15],
                                            user_id,
                                            st.session_state.loaded_data[10], 
                                            st.session_state.loaded_data[16], )  
            st.write("Top 10 recommended movies (movie_id: predicted_rating):")
            for movie_id, rating in top_movies:
                st.write(f"Movie ID: {movie_id}, Predicted Rating: {rating:.2f}")

                
# Page routing logic
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'model_selection':
    model_selection_page()
elif st.session_state.page == 'input':
    input_page()

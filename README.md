# Movie Recommendation System - Netflix Dataset
This project focuses on building a Movie Recommendation System using classical machine learning and deep learning techniques. The system predicts user ratings and recommends movies based on user preferences, leveraging collaborative filtering and matrix factorization algorithms.

## Table of Contents
*Overview
*Features
*Project Structure
*Data
*Models
*Installation
*Usage
*Future Work
*Contributing
*License

### Overview
The recommendation system utilizes both classical and advanced machine learning models to suggest the top 10 movies based on a user's preferences. We developed six models, including Collaborative Filtering (User-User and Item-Item based), Matrix Factorization (SVD), and deep learning models (ANN, ANN+Residual Connections, AutoEncoder). The Netflix dataset was used for training and evaluation.

### Features
Predicts user ratings for specific movies.
Recommends the top 10 movies tailored to the user's preferences.
Efficient handling of large datasets using sparse matrix representation.
Future enhancements: incorporating Restricted Boltzmann Machine (RBM) and a Voting technique to improve recommendations.

### Project Structure

data/
├── edited_rating.csv
├── ratings.csv
├── ratings2.csv
├── small_rating_for_item_item.csv
├── small_rating.csv
item_item_collaborative_filtering/
├── averages.pkl
├── deviations.pkl
├── movie2user.pkl
├── neighbors.pkl
├── item_item.ipynb
├── user2movie.pkl
├── usermovie2rating_test.pkl
├── usermovie2rating.pkl
matrix_factorization/
├── Atest.npz
├── Atrain.npz
├── matrix_fact.ipynb
├── matrixFactorization_ANN.ipynb
├── matrixFactorization_AutoEncoder.ipynb
├── matrixFactorization_MF_res_ANN.ipynb
├── MF_model_ANN.h5
├── MF_model_lin_res_ANN.h5
├── MF_model.h5
├── model_AutoEncoder.h5
├── mu.pkl
user_user_collaborative_filtring/
├── averages.pkl
├── deviations.pkl
├── movie2user.pkl
├── neighbors.pkl
├── user_user.ipynb
├── user2movie.pkl
├── usermovie2rating_test.pkl
├── usermovie2rating.pkl
app.py
LICENSE
README.md
requirements.txt

### Data
We used the Netflix dataset for this project.

### Data Preprocessing:
Handling missing values in the rating matrix.
Creating dictionaries (user_to_movies, movies_to_user) for collaborative filtering.
Sparse matrix representation (lil_matrix) for efficient storage in deep learning models.

### Models
#### Collaborative Filtering Models:
User-User Collaborative Filtering: Finds similar users and recommends movies they have liked.
Item-Item Collaborative Filtering: Finds similar movies to those the user has already rated highly.
#### Matrix Factorization Models:
SVD (Singular Value Decomposition): Matrix factorization technique to predict missing ratings.
ANN (Artificial Neural Network): Predicts ratings using deep learning.
ANN + Residual Connections: Enhances ANN performance using residual connections.
AutoEncoder: Learns compressed representations of user-movie interactions.

### Installation
Clone this repository:
git clone https://github.com/ahmedomer13218/RecommendationSystemV2
cd RecommendationSystemV2
Install the required dependencies:
pip install -r requirements.txt

### Usage
Run the Streamlit app to interact with the recommendation system:  
streamlit run app.py

### Future Work
Restricted Boltzmann Machine (RBM): We plan to implement RBM for enhanced recommendations.
Voting Technique: Combining the results of multiple models to improve prediction accuracy.

### Contributing
We welcome contributions! If you'd like to contribute, please fork the repository and make your changes. After testing, submit a pull request for review.

### License

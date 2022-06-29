This branch was created in order to train a model with reduced file size, for deployment onto Heroku.

As the free tier of Heroku can only hold up to 500mb of data, using the optimised Random Tree Classifier model is not possible as the model takes up massive amounts of space. Hence, for demonstration purposes, the Streamlit application is loaded using a Decision Tree Classifier model instead, as this model takes up significantly less space. Accuracy achieved with this model is 69.1%.

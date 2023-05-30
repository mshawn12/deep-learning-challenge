# deep-learning-challenge
## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
- `EIN` and `NAME` —Identification columns
- `APPLICATION_TYPE` —Alphabet Soup application type
- `AFFILIATION` —Affiliated sector of industry
- `CLASSIFICATION` —Government organization classification
- `USE_CASE` —Use case for funding
- `ORGANIZATION` —Organization type
- `STATUS` —Active status
- `INCOME_AMT` —Income classification
-`SPECIAL_CONSIDERATIONS` —Special considerations for application
- `ASK_AMT` —Funding amount requested
- `IS_SUCCESSFUL` —Was the money used effectively

## Instructions
### Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
    - What variable(s) are the target(s) for your model?
    - What variable(s) are the feature(s) for your model?

2. Drop the EIN and NAME columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

6. Use pd.get_dummies() to encode categorical variables.

7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.
    - Add more neurons to a hidden layer.
- Add more hidden layers.
- Use different activation functions for the hidden layers.
- Add or reduce the number of epochs to the training regimen.

Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

### Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

1. Overview of the analysis: Explain the purpose of this analysis.
- The purpose of this analysis is to assist the Non-Profit foundation, Alphabet Soup, by developing a tool that can help select applicants for funding with the best chance of success in their ventures. By using machine learning & neural networks, the goal of the analysis is to use various features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

2. Results: Using bulleted lists and images to support your answers, address the following questions:

- Data Preprocessing
    - What variable(s) are the target(s) for your model?
        - The target for the model was the `IS_SUCCESSFUL` variable
    - What variable(s) are the features for your model?
        - `NAME`
        - `APPLICATION_TYPE`
        - `AFFILIATION`
        - `CLASSIFICATION`
        - `USE_CASE`
        - `ORGANIZATION`
        - `INCOME_AMT`
        - `ASK_AMT`
    - What variable(s) should be removed from the input data because they are neither targets nor features?
        - `EIN`
        - `STATUS`
        - `SPECIAL_CONSIDERATIONS`
- Compiling, Training, and Evaluating the Model
    - How many neurons, layers, and activation functions did you select for your neural network model, and why?
        - The details of the final chosen optimization model are listed below. Many combinations of layers were tested with varying nodes and the below yielded the best results. 
        - (3) Hidden Layers
            - Layer 1: 66 nodes
            - Layer 2: 9 nodes
            - Layer 3: 3 nodes
        - (3) different activation functions: Tanh, Relu, Sigmoid
- Were you able to achieve the target model performance?
    - Yes, the final model performance had 78.71% accuracy. The goal was to achieve a model accuracy of over 75%
- What steps did you take in your attempts to increase model performance?
    - Adding/dropping various features to test the impact on the model
        - Ran into issues with RAM quota for free version of Colab
    - Testing multiple unique values for cutoff points to bin "rare" categorical variables together
   - Changing the features used for binning
    - Testing multiple `random_state` values in `train_test_split`
    - Adjusting the number of hidden layers. Tested having 2,3, and 4 hidden layers with various counts
    - Trying different activation methods such as relu, para_relu, selu, tanh, and more
    - Adjusting the number of epochs
    - Using `X_train_scaled` in model instead of `X_train`
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
- The initial deep learning model without any optimizations had an accuracy of about 73.04%
<img src='https://github.com/mshawn12/deep-learning-challenge/blob/main/Images/AlphabetSoup_model_eval_original.png?raw=true'>

- After making the adjustments noted above, the optimized deep learning model had an accuracy of about 78.71%
<img src='https://github.com/mshawn12/deep-learning-challenge/blob/main/Images/AlphabetSoup_model_eval_optimized.png?raw=true'>

- Another possible model to attempt with this dataset is a Logistic Regression Model. Neural Network Models are more flexible, but are also susceptible to overfitting and can be difficult to train. Overfitting was observed many times while testing various combinations of features, additional layers, neurons, activation methods, etc. Since a Logistic Regression Model is similar to Neural Networks and can be used for binary classification as well, it may be worth experimenting with. While Neural Network Models tend to be better than Logistic Regression, Logistic Regression may be worth considering in this scenario.
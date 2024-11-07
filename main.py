import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics, model_selection, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

#Test Data For Console Interaction
# 0.11,0.05,0.22,0,0.22,0.05,0,0,0.05,0.11,0.11,0.56,0.05,0,0,0.11,0.16,0,1.35,0,0.73,0,0,0,1.69,1.3,0,0.05,0,0.11,0.16,0,0.05,0,0.33,0.05,0.33,0,0,0.05,0,0.11,0,0.11,0.05,0,0,0.05,0.025,0.085,0,0.042,0,0,2.031,22,971

#Loads the dataset from the CSV file
spambase_data = 'data/spambase.csv'
df = pd.read_csv(spambase_data)

#Creates a logistical regression model object
#Had to add the max number of iterations because by default if it takes more than 100 iterations before converging, it will throw a ConvergenceWarning error
logistical_regression_model = linear_model.LogisticRegression(max_iter=4601)

#Creates a Naive Bayes model object
naive_bayes_model = MultinomialNB()

#Creates a Support Vector Machine model object
svm_model = svm.SVC(max_iter=4601)

#Used to break the data into subsets
#Dependant variable (The data we're going to predict)
y = df.values[:,57]
#Independant variable (The data we use to make a prediction)
X = df.values[:, 0:57]
#Further separates data into training and testing sets per Dr Jim Ashe tutorial
#Added the random_state parameter after researching why my accuracy results were different each time I ran my code. This ensures reproducibility by assuring my training and testing sets are the same every time
X_train, X_test, y_Train, y_Test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

#Trains the model using logistical regression - Non-Descriptive Method/Function
logistical_regression_model.fit(X_train, y_Train)

#Trains the model using the Naive Bayes Classifier - Non-Descriptive Method/Function
naive_bayes_model.fit(X_train, y_Train)

#Trains the model using a Support Vector Machine - Non-Descriptive Method/Function
svm_model.fit(X_train, y_Train)

#Checks the accuracy of the trained model using logistical regression and prints it to the console - Pre separated training/testing accuracy 0.9319713105846555 - Post separated training/testing accuracy 0.9333816075307748
#These accuracy results were from before I added the random_state parameter
y_pred = logistical_regression_model.predict(X_test)
print("Logistical Regression Accuracy:", metrics.accuracy_score(y_Test, y_pred))
#Checks and prints the accuracy of the LR model using Cross Validation to help estimate how well the model will perform on unseen data by splitting the dataset into training and testing sets 5 times (folds) - 0.92073833 0.92826087 0.9326087  0.94673913 0.83369565
print("Logistic Regression Cross-Validation Accuracy:", cross_val_score(logistical_regression_model, X, y, cv=5))

#Checks the accuracy of the trained model using the Naive Bayes Classifier - 0.782041998551774
y_pred_nb = naive_bayes_model.predict(X_test)
print("Naive Bayes Model Accuracy:", metrics.accuracy_score(y_Test, y_pred_nb))
#Checks and prints the accuracy of the NB model using Cross Validation to help estimate how well the model will perform on unseen data by splitting the dataset into training and testing sets 5 times (folds) - 0.79261672 0.81847826 0.81521739 0.78586957 0.69673913
print("Naive Bayes Cross-Validation Accuracy:", cross_val_score(naive_bayes_model, X, y, cv=5))

#Checks the accuracy of the trained model using the Support Vector Machine - 0.6799420709630702
y_pred_svm = svm_model.predict(X_test)
print("SVM Model Accuracy:", metrics.accuracy_score(y_Test, y_pred_svm))
#Checks and prints the accuracy of the SVM model using Cross Validation to help estimate how well the model will perform on unseen data by splitting the dataset into training and testing sets 5 times (folds) - 0.6558089  0.73152174 0.69891304 0.72934783 0.70978261
print("SVM Cross-Validation Accuracy:", cross_val_score(svm_model, X, y, cv=5))

#Function that uses the trained model and input data to make a prediction and print "spam" or "not spam" instead of "1" or "0" - Descriptive Method/Function
def predict_and_print_label(model, data):
    #Makes a variable named prediction and inputs data to it
    prediction = model.predict(data)

    #Maps the prediction to a label
    label = "Spam" if prediction[0] == 1 else "Not Spam"

    #Prints the prediction
    print("This email is:", label)


#Function to prompt the user for input, validate the data, and ensure correctness through a loop in the console
def get_input_data():
    while True:
        try:
            #Prompts user to input data
            print("\nPlease enter 57 feature values separated by commas, or copy and paste a row of data (the predictive data) minus the last field (the target variable)")
            input_string = input()

            #Converts the input string to a list of floats
            input_data = [float(value) for value in input_string.split(',')]

            #Checks if the input data has the correct number of features
            if len(input_data) != 57:
                print("\nThe input data must match the number of features (X) being used to predict the target variable (y) - 57 features - copy and paste a row of data or input your own data\n")
                continue

            return [input_data]

        #Checks if the input data is anything other than numbers
        except ValueError:
            print("\nInvalid input. Please enter numeric values only, separated by commas.\n")


#Function that displays the descriptive methods/visualizations
def display_visualizations():
    plt.figure(figsize=(15, 10))
    df.boxplot(rot=90)
    plt.title("Box Plot")
    plt.show()

    df.hist(bins=30, figsize=(20, 15), edgecolor='black')
    plt.suptitle("Histograms of Features")
    plt.show()

    plt.figure(figsize=(20, 15))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


#Calls the function with the model and user-input data
if __name__ == "__main__":
    #Gets input data from user
    user_input_data = get_input_data()

    #Calls the function with the logistic regression model and user-input data
    predict_and_print_label(logistical_regression_model, user_input_data)

    #Prompts the user if they want to view the descriptive methods
    while True:
        view_visuals = input("Would you like to view the data visualizations? (yes/no): ").strip().lower()
        if view_visuals == 'yes':
            display_visualizations()
            break
        elif view_visuals == 'no':
            print("Skipping visualizations")
            break
        else:
            print("Please enter 'yes' or 'no'")

# #Function that uses the trained model and input data to make a prediction and print "spam" or "not spam" instead of "1" or "0" - Non-Descriptive Method/Function
# #Used this function to test inputs and later decided to use it for user interaction via console prompts
# def predict_and_print_label(model, data):
#     #Makes a variable named prediction and inputs data to it
#     prediction = model.predict(data)
#
#     #Maps the prediction to a label
#     label = "spam" if prediction[0] == 1 else "not spam"
#
#     #Prints the prediction
#     print("Prediction:", label)
#
#
# #The input data must match the number of features (X) being used to predict the target variable (y) - 57 features - copy and paste a row of data minus the last number from the spambase.csv file or input your own data into the fields
# input_data = [[0.11,0.05,0.22,0,0.22,0.05,0,0,0.05,0.11,0.11,0.56,0.05,0,0,0.11,0.16,0,1.35,0,0.73,0,0,0,1.69,1.3,0,0.05,0,0.11,0.16,0,0.05,0,0.33,0.05,0.33,0,0,0.05,0,0.11,0,0.11,0.05,0,0,0.05,0.025,0.085,0,0.042,0,0,2.031,22,971]]
#
# #Calls the function with the model and input data
# predict_and_print_label(logistical_regression_model, input_data)
#
# #Used to test the prediction of the trained model using a 2D array per Dr Jim Ashe tutorial - Non-Descriptive Method/Function
# print(logistical_regression_model.predict([[0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.778,0,0,3.756,61,278]]))
#
#
# #Creates a box plot for visualizing the spread and central tendency of the features and identifying outliers - Descriptive Method
# df.boxplot(figsize=(15, 10), rot=90)
# plt.show()
#
# #Creates a correlation matrix with heatmap to visualize correlations between features - Descriptive Method
# corr_matrix = df.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.show()
#
# #Creates a scatter matrix to visualize the distribution of the data in the dataframe - Descriptive Method
# scatter_matrix(df)
# plt.show()
#
# #Creates a histogram to visualize the distribution of the data in the dataframe - Descriptive Method
# df.hist()
# plt.show()
#
# #Forces PyCharm to print full row and column data - Used for testing
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# print(df.head())
#
# #Prints dataset information - Used for testing
# print(df.head())
#
# #Prints dimensions of dataframe (Number of rows and columns) - Used for testing (4601 x 58)
# print(df.shape)
#
# #Specifies the row being printed - Used for testing
# row_index = 1
#
# #Prints the specified row with all columns - Used for testing
# print(df.iloc[row_index])

# #Assigns variables with string data from spambase files
# names_file_path = 'data/spambase.names'
# data_file_path = 'data/spambase.data'
#
# #Function used to read feature names from the spambase.names file and assign them as headers to the spambase.data file and then save as a new csv file
# #Used to transform and format my dataset retrieved from https://archive.ics.uci.edu/dataset/94/spambase
# #Code not needed anymore once csv file is created. Don't need the data or name files anymore either, but will keep in case of problems later
# def read_names(file_path):
#     iteration_names = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             feature_name = line.strip()
#             if feature_name:
#                 iteration_names.append(feature_name)
#     return iteration_names
#
# #Reads the feature names from the names file
# iteration_names = read_names(names_file_path)
#
# #Loads the data file into a DataFrame
# df = pd.read_csv(data_file_path, header=None, sep=',', engine='python')
#
# #Checks if the number of columns matches the number of feature names and assigns the feature names as column headers if check is ok
# if df.shape[1] == len(iteration_names):
#     df.columns = iteration_names
# else:
#     #Handles a potential mismatch by assigning column names to the available number
#     df.columns = iteration_names[:df.shape[1]]
#
# #Saves the DataFrame to a CSV file
# csv_file_path = 'data/spambase.csv'
# df.to_csv(csv_file_path, index=False)
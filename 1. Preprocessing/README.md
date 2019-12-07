## Preprocessing the dataset

### Missing data

- **Strategy 1:** Remove the observed row completely, loss of crucial information
- **Strategy 2:** Take the mean of the values of the same column and substitute
    - Instantiate the **SimpleImputer** object from **sklearn.impute**
    - Use the constructor to define the search, replace and source strategies of the Imputer object.
      - Search  -> missing_values=np.nan
      - Replace -> strategy='mean'
    - Fit this imputer to the dependent variables where missing data is present
    - Use the transform function to replace the values with the mean values as defined by the Imputer object.
  
### Categorical data

 - Encode the text-based data into distinct numbers i.e same numbers for same data.
 - Simply encoding the data doesn't help due to the difference in the numbers. (Eg. Country data will be wrong with some country having an higher number)
 - To solve this: **Dummy Encoding**
 - **Dummy Encoding** is the process of assigning as many column as the different categories and placing a value of 1 at the right column.
 - Use the Column Transformer and OneHotEncoder along with the np.array and fit_transform()
 - Encode the dependent variables also using the same technique

### Train-Test Split

-  Use the sklearn.model_selection - train_test_split to split the data into train and test sets
-  Specify the test_size and the random state in the options of the function.

### Feature Scaling

Defining the scaling technique for the different columns to remove the disparity for larger and smaller values.
- Standardisation : {x-mean(x)}/stdev(x) 
- Normalisation : {x-min(x)}/{max(x)-min(x)}

The scaling is achieved by defining a StandardScaler object in the sklearn.preprocessing library:
- Define the object sc_X
- fit_transform the X_train set
- transform the X_test set in the same range as the train set
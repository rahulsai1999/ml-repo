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
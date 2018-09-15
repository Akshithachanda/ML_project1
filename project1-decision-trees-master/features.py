import pandas as pd


def binarize_features(df, fare_threshold):
    '''
    Binarize `Sex` and `Fare` Columns of the given dataframe. 

    For the `Fare` column, values greater than the given `fare_threshold`
    should be set to 1 and 0 otherwise. For the `Sex` column, map `male` to 0
    and `female` to 1.
    '''
    train_df = pd.read_csv('data/titanic_train.csv')
    
    df['Sex'] = train_df["Sex"].map({"female":1, "male": 0})  ## Sex is a discrete feature
   # df['Fare'] = train_df["Fare"].mask(train_df["Fare"] < fare_threshold , 0) #TODO
    
    def func(x):
        if x < fare_threshold:
            return 0
        else:
            return 1
    df['Fare'] = train_df["Fare"].apply(func)

    return df.astype(int)



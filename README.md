# oilprediction
Oil prediction model (cross validation and normalization included)

STEPS TO USE MY DASHBOARD 

1# Download Dash in your terminal (mac : pip3 install dash)
2# Download the oil.csv file in the right repertory (coding environnement)
3# Run the Dashboard.py file in your console


Warning : My model was overfitted. The prediction is not good because of the overfitting. For now, I don't have enough data to fix that problem. The solution is to merge a new dataframe with additionnal datas to train the model with other datas. 
Nevertheless, you can see my crossvalidation. Among a set of linear model, the best model was the LinearRegression (std deviation approximately equal to 0.007 and mean score = 0.986

Linear Regression score (test_set) = 0.994


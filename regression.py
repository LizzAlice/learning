import pandas as pd
import quandl

# regression: take continuous data, find the equation that fits them best and make forecasts on that basis
# for example linear regression; a popular use is stock prices

df = quandl.get("WIKI/GOOGL")
print(df.head())

# here, we have too much data, as it is redundant, for example, it doesn't make any sense to keep the normal columns and the adjusted columns
# the adjusted columns are better, because they account for things like stock splits,
# where 1 share becomes 2, halving the value of the share but not the value of the company

# in a dataframe, [] selects one column and gives a pandas.Series, while [[]] can be used
# for several columns and gives a dataframe
# machine learning can only work if the data is meaningful. in stock market, according to the tutorial,
# the relationship between price changes and volume over time might be meaningful
df = df[['Adj. Open', 'Adj.High', 'Adj.Low', 'Adj.Close', 'Adj.Volume']]

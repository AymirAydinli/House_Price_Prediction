import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('boston.csv', index_col=0)

st.text('''Understand the Boston House Price Dataset
:Attribute Information (in order):
    1. CRIM     per capita crime rate by town
    2. ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS    proportion of non-retail business acres per town
    4. CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX      nitric oxides concentration (parts per 10 million)
    6. RM       average number of rooms per dwelling
    7. AGE      proportion of owner-occupied units built prior to 1940
    8. DIS      weighted distances to five Boston employment centres
    9. RAD      index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
        11. PTRATIO  pupil-teacher ratio by town
        12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        13. LSTAT    % lower status of the population
        14. PRICE     Median value of owner-occupied homes in $1000's''')


st.subheader('Preliminary Data Exploration 🔎')

st.text('data head')
st.code(data.head())
st.text('data shape:')
st.code(data.shape)

st.subheader('Data Cleaning - Check for Missing Values and Duplicates')
st.text('data.isna().any()')
st.code(data.isna().any())
st.text('data.duplicated().any()')
st.code(data.duplicated().any())

st.subheader('Descriptive Statistics')
st.code(data.describe())

st.text('''On average 18 students per teacher

Average price of a home is 23k$

Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

Min Number of rooms is 4 and Max is 9''')


st.subheader('Visualise the Features')


st.text('House Prices 💰')
sns.displot(data['PRICE'],
            bins=50,
            aspect=2,
            kde=True,
            color='#2196f3')

plt.title(f'1970s Home Values in Boston. Average: ${(1000*data.PRICE.mean()):.6}')
plt.xlabel('Price in 000s')
plt.ylabel('Nr. of Homes')

st.pyplot(plt)

st.text('Distance to Employment - Length of Commute')

sns.displot(data['DIS'],
            bins=50,
            aspect=2,
            kde=True,
            color='darkblue')

plt.title(f'Distance to Employment Centres. Average: ${(1000*data.DIS.mean()):.6}')
plt.xlabel('Weighted Distance to 5 Boston Employment Centres')
plt.ylabel('Nr. of Homes')

st.pyplot(plt)


st.text('Number of Rooms')
sns.displot(data.RM,
            aspect=2,
            kde=True,
            color='#00796b')

plt.title(f'Distribution of Rooms in Boston. Average: {data.RM.mean():.2}')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Nr. of Homes')

st.pyplot(plt)

st.text('Access to Highways')

plt.figure(figsize=(10, 5), dpi=200)

plt.hist(data['RAD'],
         bins=24,
         ec='black',
         color='#7b1fa2',
         rwidth=0.5)

plt.xlabel('Accessibility to Highways')
plt.ylabel('Nr. of Houses')
st.pyplot(plt)

st.text('Next to the River? ⛵️')

river_access = data['CHAS'].value_counts()

bar = px.bar(x=['No', 'Yes'],
             y=river_access.values,
             color=river_access.values,
             color_continuous_scale=px.colors.sequential.haline,
             title='Next to Charles River?')

bar.update_layout(xaxis_title='Property Located Next to the River?',
                  yaxis_title='Number of Homes',
                  coloraxis_showscale=False)
st.plotly_chart(bar)

st.text('Distance from Employment vs Pollution')

with sns.axes_style('darkgrid'):
  sns.jointplot(x=data['DIS'],
                y=data['NOX'],
                height=8,
                kind='scatter',
                color='deeppink',
                joint_kws={'alpha':0.5})

st.pyplot(plt)

st.text('We see that pollution goes down as we go further and further out of town.'
        ' This makes intuitive sense. However, even at the same distance of 2 miles'
        ' to employment centres, we can get very different levels of pollution.'
        ' By the same token, DIS of 9 miles and 12 miles have very similar levels of pollution.')


st.text('Proportion of Non-Retail Industry 🏭🏭🏭 versus Pollution')

with sns.axes_style('darkgrid'):
  sns.jointplot(x=data.NOX,
                y=data.INDUS,
                # kind='hex',
                height=7,
                color='darkgreen',
                joint_kws={'alpha':0.5})
st.pyplot(plt)

st.text('% of Lower Income Population vs Average Number of Rooms')

st.text('In the top left corner we see that all the homes with 8 or more rooms,'
        ' LSTAT is well below 10%')

st.header('Split Training & Test Dataset')

target = data['PRICE']
features = data.drop('PRICE', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=10)
st.text('% of training set')
train_pct = 100*len(X_train)/len(features)
st.code(f'Training data is {train_pct:.3}% of the total data.')

st.text('% of test data set')
test_pct = 100*X_test.shape[0]/features.shape[0]
st.code(f'Test data makes up the remaining {test_pct:0.3}%.')


st.text('''Multivariable Regression¶
Our Linear Regression model will have the following form:

𝑃𝑅𝐼̂ 𝐶𝐸=𝜃0+𝜃1𝑅𝑀+𝜃2𝑁𝑂𝑋+𝜃3𝐷𝐼𝑆+𝜃4𝐶𝐻𝐴𝑆...+𝜃13𝐿𝑆𝑇𝐴𝑇''')

st.text('Run Your Regression')
st.text('''regr = LinearRegression()
regr.fit(X_train, y_train)
rsquared = regr.score(X_train, y_train)''')
regr = LinearRegression()
regr.fit(X_train, y_train)
rsquared = regr.score(X_train, y_train)

st.code(f'Training data r-squared: {rsquared:.2}')

st.text('0.75 is a very high r-squared!')

st.text('Evaluate the Coefficients of the Model')

regr_coef = pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['Coefficient'])
st.code(regr_coef)

st.subheader('Analyse the Estimated Values & Regression Residuals')
st.text('''The next step is to evaluate our regression. How good our regression is depends not only on the r-squared. It also depends on the residuals - the difference between the model's predictions ( 𝑦̂ 𝑖
 ) and the true values ( 𝑦𝑖
 ) inside y_train.''')

predicted_vals = regr.predict(X_train)
residuals = (y_train - predicted_vals)
st.text('''predicted_vals = regr.predict(X_train)
residuals = (y_train - predicted_vals)''')

st.text('Original Regression of Actual vs. Predicted Prices')
plt.figure(dpi=100)
plt.scatter(x=y_train, y=predicted_vals, c='indigo', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.title(f'Actual vs Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
st.pyplot(plt)

st.text('Residuals vs Predicted values')
plt.figure(dpi=100)
plt.scatter(x=predicted_vals, y=residuals, c='indigo', alpha=0.6)
plt.title('Residuals vs Predicted Values', fontsize=17)
plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
st.pyplot(plt)

st.text('''Why do we want to look at the residuals?
We want to check that they look random. Why?
The residuals represent the errors of our model. 
If there's a pattern in our errors, then our model has a systematic bias.

We can analyse the distribution of the residuals. 
In particular, we're interested in the skew and the mean.

In an ideal case, what we want something close to a normal distribution.
A normal distribution has a skewness of 0 and a mean of 0.
A skew of 0 means that the distribution is symmetrical - the bell curve is not lopsided or biased to one side. 
Here's what a normal distribution looks like:''')

st.text('Residual Distribution Chart')
resid_mean = round(residuals.mean(), 2)
resid_skew = round(residuals.skew(), 2)

sns.displot(residuals, kde=True, color='indigo')
plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')
st.pyplot(plt)

st.text('We see that the residuals have a skewness of 1.46. There could be some room for improvement here.')

st.header('Data Transformations for a Better Fit')
st.text("Let's try a data transformation approach.")

st.text("Investigate if the target data['PRICE'] could be a suitable candidate for a log transformation.")

tgt_skew = data['PRICE'].skew()
sns.displot(data['PRICE'], kde='kde', color='green')
plt.title(f'Normal Prices. Skew is {tgt_skew:.3}')
st.pyplot(plt)

y_log = np.log(data['PRICE'])
sns.displot(y_log, kde=True)
plt.title(f'Log Prices. Skew is {y_log.skew():.3}')
st.pyplot(plt)

st.text('''The log prices have a skew that's closer to zero. 
This makes them a good candidate for use in our linear model. 
Perhaps using log prices will improve our regression's r-squared and 
our model's residuals.''')


st.subheader('How does the log transformation work?')

st.text('''Using a log transformation does not affect every price equally. 
Large prices are affected more than smaller prices in the dataset. 
Here's how the prices are "compressed" by the log transformation:''')


plt.figure(dpi=150)
plt.scatter(data.PRICE, np.log(data.PRICE))

plt.title('Mapping the Original Price to a Log Price')
plt.ylabel('Log Price')
plt.xlabel('Actual $ Price in 000s')
st.pyplot(plt)

st.header('Regression using Log Prices')

new_target = np.log(data['PRICE']) # Use log prices
features = data.drop('PRICE', axis=1)

X_train, X_test, log_y_train, log_y_test = train_test_split(features,
                                                    new_target,
                                                    test_size=0.2,
                                                    random_state=10)

log_regr = LinearRegression()
log_regr.fit(X_train, log_y_train)
log_rsquared = log_regr.score(X_train, log_y_train)

log_predictions = log_regr.predict(X_train)
log_residuals = (log_y_train - log_predictions)

st.code(f'Training data r-squared: {log_rsquared:.2}')
st.text("This time we got an r-squared of 0.79 compared to 0.75. This looks like a promising improvement.")

st.header('Evaluating Coefficients with Log Prices')

df_coef = pd.DataFrame(data=log_regr.coef_, index=X_train.columns, columns=['coef'])
st.code(df_coef)

st.text('''The key thing we look for is still the sign - being close to the river 
results in higher property prices because CHAS has a coefficient greater than zero. 
Therefore property prices are higher next to the river.

More students per teacher - a higher PTRATIO - is a clear negative. 
Smaller classroom sizes are indicative of higher quality education, 
so have a negative coefficient for PTRATIO.''')

st.text('Distribution of Residuals (log prices) - checking for normality')
log_resid_mean = round(log_residuals.mean(), 2)
log_resid_skew = round(log_residuals.skew(), 2)

sns.displot(log_residuals, kde=True, color='navy')
plt.title(f'Log price model: Residuals Skew ({log_resid_skew}) Mean ({log_resid_mean})')
st.pyplot(plt)

sns.displot(residuals, kde=True, color='indigo')
plt.title(f'Original model: Residuals Skew ({resid_skew}) Mean ({resid_mean})')
st.pyplot(plt)

st.text('''Our new regression residuals have a skew of 0.09 compared to a skew of 1.46. 
The mean is still around 0. 
From both a residuals perspective and an r-squared perspective 
we have improved our model with the data transformation.''')
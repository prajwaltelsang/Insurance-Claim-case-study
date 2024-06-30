#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats  #for the statistical tests
from scipy import stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


claims = pd.read_csv('claims.csv')


# In[3]:


cust = pd.read_csv('cust_demographics.csv')


# In[4]:


claims.head()


# In[5]:


cust.head()


# 1. Import claims_data.csv and cust_data.csv which is provided to you and 
# combine the two datasets appropriately to create a 360-degree view of 
# the data. Use the same for the subsequent questions.

# In[6]:


insurance_claim = pd.merge(left = claims, right = cust, left_on = 'customer_id', right_on = 'CUST_ID', how = 'outer')
insurance_claim.head()


# In[7]:


insurance_claim.drop(columns ='CUST_ID', inplace = True)


# 2. Perform a data audit for the datatypes and find out if there are any 
# mismatch within the current datatypes of the columns and their 
# business significance.

# In[8]:


insurance_claim.info()


# In[9]:


insurance_claim.claim_date = pd.to_datetime(insurance_claim['claim_date'], format = "%m/%d/%Y")


# In[10]:


insurance_claim['DateOfBirth'] = pd.to_datetime(insurance_claim['DateOfBirth'], format = "%d-%b-%y")


# In[11]:


insurance_claim['DateOfBirth'] = insurance_claim.DateOfBirth.apply( lambda x : x - pd.DateOffset(years = 100) if x.year > 2000 else x)


# In[12]:


insurance_claim['total_policy_claims'] = insurance_claim['total_policy_claims'].astype('object')


# 3. Convert the column claim_amount to numeric. Use the appropriate 
# modules/attributes to remove the $ sign.

# In[13]:


insurance_claim.claim_amount = insurance_claim['claim_amount'].str.replace("$", "")


# In[14]:


insurance_claim.claim_amount = insurance_claim['claim_amount'].astype("float")


# In[15]:


insurance_claim.info()


# 4. Of all the injury claims, some of them have gone unreported with the 
# police. Create an alert flag (1,0) for all such claims.

# In[16]:


insurance_claim.claim_type.unique()


# In[17]:


insurance_claim['alert_flag'] = np.where((insurance_claim.claim_type.str.contains('injury', case = False)) & (insurance_claim.police_report == 'No'), 1 , 0)


# In[18]:


insurance_claim.police_report.unique()


# 5. One customer can claim for insurance more than once and in each claim,
# multiple categories of claims can be involved. However, customer ID 
# should remain unique. 

# In[19]:


insurance_claim.sort_values('claim_date', ascending = False, inplace = True)
insurance_claim.drop_duplicates(subset = ['customer_id'], keep = 'first', inplace = True)


# In[20]:


insurance_claim.shape


# 6. Check for missing values and impute the missing values with an 
# appropriate value. (mean for continuous and mode for categorical

# In[21]:


insurance_claim.isna().sum()


# In[22]:


continuous_col = insurance_claim.select_dtypes(include = ["number"]).columns
continuous_col


# In[23]:


categorical_col = insurance_claim.select_dtypes(include = ["object", 'datetime']).columns
categorical_col


# In[24]:


for col in continuous_col:
    insurance_claim[col].fillna(insurance_claim[col].mean(), inplace = True)


# In[25]:


for col in categorical_col:
    insurance_claim[col].fillna(insurance_claim[col].mode()[0], inplace = True)


# In[26]:


insurance_claim.isna().sum()


# 7. Calculate the age of customers in years. Based on the age, categorize the
# customers according to the below criteria
# Children < 18
# Youth 18-30
# Adult 30-60
# Senior > 60

# In[27]:


insurance_claim['age'] = (pd.Timestamp.now() - insurance_claim['DateOfBirth']).dt.days // 365


# In[28]:


bins = [0, 18, 30, 60, 100]
labels = ['Children', 'Youth', 'Adult', 'Senior']
insurance_claim['age_category'] = pd.cut(insurance_claim['age'], bins = bins, labels = labels)


# In[29]:


insurance_claim.head()


# 8. What is the average amount claimed by the customers from various 
# segments?

# In[30]:


insurance_claim.groupby('Segment')[['claim_amount']].mean()


# 9. What is the total claim amount based on incident cause for all the claims
# that have been done at least 20 days prior to 1st of October, 2018.

# In[31]:


filtered_claim = insurance_claim[(pd.to_datetime('2018-10-01') - insurance_claim['claim_date']).dt.days >= 20]


# In[32]:


filtered_claim.groupby("incident_cause")[['claim_amount']].sum()


# 10. How many adults from TX, DE and AK claimed insurance for driver 
# related issues and causes?

# In[33]:


insurance_claim.loc[(insurance_claim['age_category'] == 'Adult') & 
                             (insurance_claim['State'].isin(['TX', 'DE', 'AK'])) &
                             (insurance_claim['incident_cause'].str.contains('Driver', case = False))].groupby('State')[['age_category']].count()


# 11. Draw a pie chart between the aggregated value of claim amount based 
# on gender and segment. Represent the claim amount as a percentage on
# the pie chart.
# 

# In[34]:


pivot_table = insurance_claim.pivot_table(index = 'gender', columns = 'Segment', values = 'claim_amount', aggfunc = 'sum')
pivot_table.plot(kind = 'pie', subplots = True, figsize = (15,8), autopct = "%1.1f%%")
plt.show()


# 12. Among males and females, which gender had claimed the most for any 
# type of driver related issues? E.g. This metric can be compared using a 
# bar chart

# In[35]:


insurance_claim.loc[insurance_claim.incident_cause.str.contains("driver", case = False)].groupby("gender")[['claim_amount']].sum().plot(kind = 'bar', color = 'magenta')
plt.ylabel("claim amount")
plt.show()


# 13. Which age group had the maximum fraudulent policy claims? Visualize 
# it on a bar chart.
# 

# In[36]:


insurance_claim.loc[insurance_claim.fraudulent == 'Yes'].groupby("age_category")[['fraudulent']].count().plot(kind = 'bar', color = 'blue')
plt.show()


# 14. Visualize the monthly trend of the total amount that has been claimed 
# by the customers. Ensure that on the “month” axis, the month is in a 
# chronological order not alphabetical order. 

# In[37]:


insurance_claim['month'] = insurance_claim['claim_date'].dt.month

# Group the data by month and calculate the sum of claim_amount
monthly_claim_amount = insurance_claim.groupby('month')['claim_amount'].sum()

# Sort the data by month in chronological order
monthly_claim_amount = monthly_claim_amount.sort_index()
import calendar
month_names = [calendar.month_name[i] for i in range(1, 13)]
# line plot using the monthly_claim_amount data
plt.figure(figsize = (16,8))
plt.plot(monthly_claim_amount.index, monthly_claim_amount.values)
plt.xlabel('Month')
plt.ylabel('Total Claim Amount')
plt.title('Monthly Trend of Total Claim Amount')
plt.xticks(range(1, 13), month_names)
plt.show()


# 15. What is the average claim amount for gender and age categories and 
# suitably represent the above using a facetted bar chart, one facet that 
# represents fraudulent claims and the other for non-fraudulent claims.

# In[38]:


avg_claim_amount = insurance_claim.groupby(['gender', 'age_category', 'fraudulent'])['claim_amount'].mean().reset_index()

# Create a facetted bar chart
g = sns.catplot(x='age_category', y='claim_amount', hue = 'gender', col = 'fraudulent', data = avg_claim_amount, kind = 'bar', height = 4, aspect = .7)
g.set_axis_labels('Age Category', 'Average Claim Amount')
plt.show()


# Based on the conclusions from exploratory analysis as well as suitable 
# statistical tests, answer the below questions. Please include a detailed 
# write-up on the parameters taken into consideration, the Hypothesis 
# testing steps, conclusion from the p-values and the business implications of 
# the statements. 

# 16. Is there any similarity in the amount claimed by males and females?

# In[39]:


## To test whether there is any similarity in the amount claimed by males and females, we can use a two-sample t-test.

## Step 1: Defining the null and alternate hypothesis - 

## H0: There is no difference in the mean amount claimed by males and females.
## Ha: There is a difference in the mean amount claimed by males and females.

## Step 2: set the level of significance - 
## CI = 95%
## alpha - 0.05


# In[40]:


sns.boxplot(x = insurance_claim.gender, y = insurance_claim.claim_amount )
plt.show()


# In[41]:


## Step 3: conduct the two sample t-test:
import scipy.stats as stats
male_claim = insurance_claim[insurance_claim['gender'] == 'Male']['claim_amount']
female_claim = insurance_claim[insurance_claim['gender'] == 'Female']['claim_amount']


# In[42]:


stats.ttest_ind(male_claim, female_claim)


# In[43]:


## we can also perform ANOVA :

stats.f_oneway(male_claim, female_claim)


# Conclusion:
# The p-value obtained from the test is greater than the level of significance (alpha=0.05), which suggests that we fail to reject the null hypothesis. Therefore, we can conclude that there is not enough evidence to suggest a difference in the mean amount claimed by males and females.¶
# From a business perspective, this result may suggest that gender is not a significant factor in determining the amount claimed by customers. However, it is important to note that there may be other factors that can influence the amount claimed, such as age, occupation, etc. Further analysis may be required to understand the impact of these factors on the amount claimed.

# 17. Is there any relationship between age category and segment?

# In[44]:


## To test if there is any relationship between age category and segment, we can use a chi-square test of independence.

## Step 1: Defining the null and alternate hypothesis - 

## H0: there is no relationship between age category and segment.
## Ha: there is an relationship between age category and segment.

## Step 2: set the level of significance - 
## CI = 95%
## alpha - 0.05


# In[45]:


## Step 3: Create a contingency table of observed frequencies of age category and segment.

observed_freq = pd.crosstab(index = insurance_claim.Segment, columns = insurance_claim.age_category)
observed_freq


# In[46]:


## Step 4: perform the chi-square test:
stats.chi2_contingency(observed_freq)


# The result of the chi-square test includes three values: the chi-square statistic, the p-value, and the degrees of freedom, as well as the expected frequency counts.

# In the given result, the chi-square statistic value is 5.3589, which indicates the strength of the relationship between the age category and segment. The p-value is 0.2524, which is greater than the standard significance level of 0.05, suggesting that we fail to reject the null hypothesis, and there is no significant relationship between the age category and segment. The degrees of freedom are 4, and the expected frequency counts are given in the array.

# In summary, the chi-square test suggests that there is no significant relationship between age category and segment. Therefore, we can conclude that there is no evidence to suggest that age category influences the choice of segment for insurance claims.

# 18.  The current year has shown a significant rise in claim amounts as 
# compared to 2016-17 fiscal average which was $10,000 ?

# In[47]:


insurance_claim.claim_date.dt.year.unique()


# In[48]:


insurance_claim.head()


# In[49]:


insurance_claim.loc[insurance_claim.claim_date.dt.year == 2017]


# In[50]:


## we need to perform t-test to check this hypothesis.
## H0: The current year's claim amount is not significantly different from the 2016-17 fiscal average of $10,000.
## Ha: The current year's claim amount is significantly different from the 2016-17 fiscal average of $10,000.
## CI = 95%
## p-value = 0.05


# In[51]:


claim_2017 = insurance_claim.loc[insurance_claim.claim_date.dt.year == 2017, 'claim_amount']
claim_2017.mean()


# In[52]:


claim_2018 = insurance_claim.loc[insurance_claim.claim_date.dt.year == 2018, 'claim_amount']
claim_2018.mean()


# In[53]:


stats.ttest_1samp(claim_2018,10000 )


# In[54]:


sns.boxplot(x = insurance_claim.claim_date.dt.year , y = insurance_claim.claim_amount)
plt.show()


# Here the p-value is 1.199e-05 (very small). Therefore, we can reject the null hypothesis that the mean claim amount for the current year is equal to the 2016-17 fiscal average, and conclude that the current year has indeed shown a significant rise in claim amounts.

# 19. Is there any difference between age groups and insurance claims?

# In[55]:


insurance_claim.age_category.value_counts()


# In[56]:


sns.boxplot(x = insurance_claim.age_category, y = insurance_claim.claim_amount)
plt.show()


# In[57]:


sns.barplot(x = insurance_claim.age_category, y = insurance_claim.claim_amount)
plt.show()


# In[58]:


## Here to test this we will perform f-test/ANOVA

## Ho : There is no significant difference between age groups and insurance claims.
## Ha : There is a significant difference between age groups and insurance claims.

## CI = 95%
## p critical = 0.05


# In[59]:


## Children claim amount is zero, hence we will not use this age_category to test the ANOVA. 
youth = insurance_claim.loc[insurance_claim.age_category == 'Youth', 'claim_amount']
Adult = insurance_claim.loc[insurance_claim.age_category == 'Adult', 'claim_amount']
Senior = insurance_claim.loc[insurance_claim.age_category == 'Senior', 'claim_amount']


# In[60]:


youth.mean()


# In[61]:


Adult.mean()


# In[62]:


Senior.mean()


# In[63]:


stats.f_oneway(youth, Adult, Senior)


# Here the p-value is greater than the significance level (α=0.05) which suggests that there is no significant difference between the mean claim amounts of different age categories. Therefore, we fail to reject the null hypothesis.

# 20. . Is there any relationship between total number of policy claims and the 
# claimed amount?
# 

# In[64]:


insurance_claim.total_policy_claims.value_counts()


# In[65]:


insurance_claim.total_policy_claims.unique()


# In[66]:


sns.boxplot(x = insurance_claim.total_policy_claims, y = insurance_claim.claim_amount)
plt.show()


# In[67]:


# Drop non-numeric columns
numeric_data = insurance_claim.select_dtypes(include=[np.number])

# Compute correlation matrix
correlation_matrix = numeric_data.corr()

# Plot heatmap
sns.heatmap(correlation_matrix, annot=True)
plt.show()


# ## To test this we can perform Pearson correlation test or ANOVA.
# ## Ho : There is no relationship between the total number of policy claims and the claimed amount.
# 
# ## Ha : There is a relationship between the total number of policy claims and the claimed amount.
# 
# ## CI = 95%
# 
# ## p-critical = 0.05

# In[68]:


## Pearson correlation test:

stats.pearsonr(insurance_claim.total_policy_claims, insurance_claim.claim_amount)


# In[69]:


## perform ANOVA

s1 = insurance_claim.loc[insurance_claim.total_policy_claims == 1, 'claim_amount']
s2 = insurance_claim.loc[insurance_claim.total_policy_claims == 2, 'claim_amount']
s3 = insurance_claim.loc[insurance_claim.total_policy_claims == 3, 'claim_amount']
s4 = insurance_claim.loc[insurance_claim.total_policy_claims == 4, 'claim_amount']
s5 = insurance_claim.loc[insurance_claim.total_policy_claims == 5, 'claim_amount']
s6 = insurance_claim.loc[insurance_claim.total_policy_claims == 6, 'claim_amount']
s7 = insurance_claim.loc[insurance_claim.total_policy_claims == 7, 'claim_amount']
s8 = insurance_claim.loc[insurance_claim.total_policy_claims == 8, 'claim_amount']


# In[70]:


stats.f_oneway(s1,s2,s3,s4,s5,s6,s7,s8)


# From Pearson correlation test, it is negative which shows there is a weak negative correlation (-0.024) between the total number of policy claims and the claimed amount. However, the p-value (0.420) is greater than the significance level of 0.05, indicating that there is not enough evidence to reject the null hypothesis. Therefore, we can conclude that there is no significant relationship between the total number of policy claims and the claimed amount.

# In[ ]:





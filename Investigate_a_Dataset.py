
# coding: utf-8

# 
# # Project: Investigate FBI Gun Data
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# In this project, we'll be exploring FBI Gun Data as well as US Census data and looking to see if there may be any correlations between the two datasets.
# 
# #### Questions to Investigate
# - What census data is most associated with high gun permits per capita?
# - Which states have had the highest growth in gun registrations?
# - What is the overall trend of gun purchases?
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling

# ### FBI Gun Data Notes
# - I will likely want to split the month column into year and month columns to group the data more usefully
# - drop all but year, month, state, & totals (other data is largely incomplete)
# 
# #### Cleaned FBI Gun Data Notes
# - Columns
#     - Year
#     - Month
#     - State: 50 states only (dropped territories and DC)
#     - Total Permits Issued: only kept the total column as it was the only column (other than month and state) with complete data

# ### Inspection

# In[2]:


# Inspect FBI Gun Data
fbi_df = pd.read_csv('gun_data.csv')
fbi_df.head()


# In[3]:


fbi_df.shape


# In[4]:


fbi_df.describe()


# In[5]:


fbi_df.info()


# ## Cleaning FBI Data

# In[6]:


# Drop unneeded columns
fbi_df.drop(['permit','permit_recheck','handgun','long_gun','other','multiple','admin','prepawn_handgun','prepawn_long_gun','prepawn_other','redemption_handgun','redemption_long_gun','redemption_other','returned_handgun','returned_long_gun','returned_other','rentals_handgun','rentals_long_gun','private_sale_handgun','private_sale_long_gun','private_sale_other','return_to_seller_handgun','return_to_seller_long_gun','return_to_seller_other'], axis=1, inplace=True)


# In[7]:


# Confirm changes
fbi_df


# In[8]:


# States to drop
row_drop = ['District of Columbia','Guam','Mariana Islands','Puerto Rico','Virgin Islands']


# In[9]:


# Query desired rows, filter out unneeded rows
fbi_df = fbi_df.query('state not in @row_drop')


# In[10]:


# Confirm changes
fbi_df


# In[11]:


# Convert month to datetime
fbi_df['date'] = pd.to_datetime(fbi_df['month'], format="%Y %m %d")


# In[12]:


# Check data types; confirm datetime data type
fbi_df.info()


# In[13]:


# Drop month column
fbi_df.drop(['month'], axis=1, inplace=True)


# In[14]:


# Split date into year and month
fbi_df['year'], fbi_df['month'] = fbi_df['date'].dt.year, fbi_df['date'].dt.month


# In[15]:


# Confirm changes
fbi_df.head()


# In[16]:


# Drop date column
fbi_df.drop(['date'], axis=1, inplace=True)


# In[17]:


# List column names
cols = fbi_df.columns.tolist()
cols


# In[18]:


# Rearrange columns
cols = cols[-2:] + cols[:2]
cols


# In[19]:


# Reorder columns in dataframe
fbi_df = fbi_df[cols]
fbi_df.head()


# In[20]:


# Confirm creation of new frame
fbi_df


# In[21]:


# Check data types
fbi_df.info()


# In[22]:


# Save cleaned data to new file
fbi_df.to_csv('gun_data_cleaned.csv', index=False)


# #### Summary: Cleaned FBI Gun Data Notes
# - Columns
#     - Year, int
#     - Month, int
#     - State, string: 50 states only (dropped territories and DC)
#     - Total Permits Issued, int: only kept the total column as it was the only column (other than month and state) with complete data
# 
# ## Cleaning Census Data

# In[23]:


# Read in Census data
census_df = pd.read_csv('U.S. Census Data.csv', thousands=',')
census_df.head()


# In[24]:


# Inspect Census shape
census_df.shape


# In[25]:


# Inspect Census info
census_df.info()


# In[26]:


# Transpose Index and Columns
census_df = census_df.T


# In[27]:


# Confirm transpose
census_df.head()


# In[28]:


# Assign Fact row as column header
census_df.columns = census_df.iloc[0]


# In[29]:


# check column labels
census_df.columns


# In[30]:


# check index labels
census_df.index


# In[31]:


# Drop original Fact and Fact Note rows
census_df.drop(['Fact','Fact Note'], axis=0, inplace=True)


# In[32]:


# check index labels
census_df.index


# In[33]:


# Rename index
census_df.index.names = ['state']


# In[34]:


# check rename
census_df.index


# In[35]:


# check column labels
census_df.columns


# In[36]:


# Drop columns
census_df.drop(census_df.loc[:,'FIPS Code':'Z'], inplace=True, axis=1)


# In[37]:


# Confirm drops
census_df.info()


# In[38]:


# remove non-numeric characters from data, convert to float, coerce other to NaN
all_columns = census_df.columns
for column in all_columns:
    census_df[column].replace(regex=True,inplace=True,to_replace=r'\D+',value=r'')
    census_df[column]=pd.to_numeric(census_df[column], downcast='float', errors='coerce')


# In[39]:


# confirm conversion
census_df.info()


# In[40]:


# check for NaNs
census_df.isnull().sum().sum()


# In[41]:


# fill NaNs with mean
census_df.fillna(census_df.mean,axis=1,inplace=True)


# In[42]:


# check NaNs again
census_df.isnull().sum().sum()


# In[43]:


# Save cleaned data to file
census_df.to_csv('census_cleaned.csv')


# #### Summary: Cleaned Census Data
# - transformed rows to columns
# - dropped Fact and Fact Note rows
# - converted objects/strings to floats
# - removed non-numeric characters from all data
# - replaced NaNs with mean

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# ### Research Question 1: Which states have had the highest growth in gun registrations?

# In[44]:


# Pull data header
fbi_df.head()


# In[45]:


# Sort ascending by state, year, and month
fbi_df = fbi_df.sort_values(['state','year','month'], ascending=True)


# In[46]:


# create column showing difference between totals between years, by state
fbi_df['diff'] = fbi_df['totals'].diff(1)
fbi_df


# In[47]:


# new frame for years 2000 through 2016
growth_10_16 = fbi_df.query('year >= 2000 and year <=2016')


# In[48]:


# list top 5 states with largest difference in gun permits between 1998 and 2016
top5 = growth_10_16.groupby('state')['diff'].sum().nlargest(5)
top5


# In[49]:


# plot of top 5 states by number of gun permits issued across all 50 states between 2000 and 2016
plt.figure(figsize=[10,10])
plt.xlabel('State')
plt.ylabel('Total # of Gun Permits Issued')
plt.title('Top 5 States issuing gun permits from 2010-2016')
top5.plot(kind='bar');


# Kentucky had the largest increase (and more than twice as many as the second largest increase) in background checks with a change from 2010 to 2016 of 361,651. California, Florida, Illinois, and Texas, round out the top 5.

# ### Research Question 2: What is the overall trend of gun purchases?

# In[50]:


g = growth_10_16.groupby(['year'])['totals'].sum().plot(x='year', y='totals', figsize=[10,10])
plt.xlabel('Year')
plt.ylabel('# of Checks Run (10\'s of millions)')
plt.title('Total # of Gun-related Background Checks Issued per year, 50 states')
plt.show();


# In[51]:


# overall growth in gun permits
change_2016 = growth_10_16.query('year == 2016')['totals'].sum()
change_2010 = growth_10_16.query('year == 2010')['totals'].sum()
o_growth = (change_2016 - change_2010) / change_2010
change_2016, change_2010, o_growth


# The overall trend for gun purchases, based on the FBI gun permit data, is positive - a change of 92.51% from 2010 to 2016.

# ### Research Questions 3: What census data is most associated with high gun permits per capita?

# In[52]:


# show census data
census_df.head()


# In[53]:


# percent population change in Kentucky between 2010 and 2016
census_df.loc['Kentucky','Population, percent change - April 1, 2010 (estimates base) to July 1, 2016,  (V2016)']


# In[54]:


# percent change in gun permits in Kentucky between 2010 and 2016
ky_2010 = fbi_df.query('year == 2010 and state == "Kentucky"')['totals'].sum()
ky_2016 = fbi_df.query('year == 2016 and state == "Kentucky"')['totals'].sum()
ky_pct_change = (ky_2016 - ky_2010) / ky_2010
(ky_pct_change*100)


# Between 2016 and 2010, Kentucky had a 54% increase in background checks for guns. As previously mentioned, this is due to their policy of running monthly checks for those with concealed carry permits.

# In[55]:


# 2010 totals by state
g10 = growth_10_16[growth_10_16['year'] == 2010].groupby('state')['totals'].sum().reset_index()
g10.rename(columns={'totals':'2010_guns'}, inplace=True)
g10.set_index('state', inplace=True)


# In[56]:


# 2016 totals by state
g16 = growth_10_16[growth_10_16['year'] == 2016].groupby('state')['totals'].sum().reset_index()
g16.rename(columns={'totals':'2016_guns'}, inplace=True)
g16.set_index('state', inplace=True)


# In[57]:


# add gun checks totals columns to census_df
census_df['2016_guns'] = g16['2016_guns']
census_df['2010_guns'] = g10['2010_guns']


# In[58]:


# add gun checks per capita columns to census_df
census_df['2016_guns_per_cap'] = census_df['2016_guns']/census_df['Population estimates, July 1, 2016,  (V2016)']
census_df['2010_guns_per_cap'] = census_df['2010_guns']/census_df['Population estimates base, April 1, 2010,  (V2016)']


# In[59]:


# confirm new columns
census_df.head()


# In[60]:


# show 5 largest gun check requests per capita for 2010 and 2016
gpc_2010 = census_df['2010_guns_per_cap'].nlargest(5)
gpc_2016 = census_df['2016_guns_per_cap'].nlargest(5)
gpc_2010, gpc_2016


# Again, Kentucky appears to be an outlier in this dataset as a result of their policy to run monthly background checks, especially on those with concealed carry permits.

# In[61]:


# scatter plot of population per square mile (2010) vs 2010 gun checks issued
census_df.plot(x='Population per square mile, 2010',y='2010_guns_per_cap', kind='scatter', figsize=[10,10])
plt.xlabel('Population per Square Mile, 2010')
plt.ylabel('Guns per capita, 2010')
plt.title('Population per square mile vs. Guns per capita, 2010');


# Other than a few outliers, it appears that there is a weak correlation between population per square mile and total gun permits issued. This would make sense as people who live in more sparsely populated areas often need guns for hunting, protecting livestock, and personal safety (in absence of a robust law enforcement presence).

# In[62]:


# plot White population vs guns per capita
census_df.plot(x='White alone, not Hispanic or Latino, percent, July 1, 2016,  (V2016)', y='2016_guns_per_cap', kind='scatter')
plt.xlabel('White alone, not Hispanic or Latino, percent (2016)')
plt.ylabel('Guns per capita, 2016')
plt.title('White population (percent, not Hispanic or Latino) vs. Guns per capita, 2016');


# In[63]:


# plot Black population vs guns per capita
census_df.plot(x='Black or African American alone, percent, July 1, 2016,  (V2016)', y='2016_guns_per_cap', kind='scatter')
plt.xlabel('Black or African American alone, percent (2016)')
plt.ylabel('Guns per capita, 2016')
plt.title('Black population (percent) vs. Guns per capita, 2016');


# In[64]:


# plot Asian population vs guns per capita
census_df.plot(x='Asian alone, percent, July 1, 2016,  (V2016)', y='2016_guns_per_cap', kind='scatter')
plt.xlabel('Asian population, percent (2016)')
plt.ylabel('Guns per capita, 2016')
plt.title('Asian population, percent vs. Guns per capita, 2010');


# In[65]:


# plot college degree holders vs guns per capita
census_df.plot(x='Bachelor\'s degree or higher, percent of persons age 25 years+, 2011-2015', y='2016_guns_per_cap', kind='scatter')
plt.xlabel('College-educated adults, percent (2011-2015)')
plt.ylabel('Guns per capita, 2016')
plt.title('College-educated adults, percent vs. Guns per capita, 2016');


# In[66]:


# plot seniors vs guns per capita
census_df.plot(x='Persons 65 years and over, percent,  July 1, 2016,  (V2016)', y='2016_guns_per_cap', kind='scatter')
plt.xlabel('Persons 65 years and over, percent (2016)')
plt.ylabel('Guns per capita, 2016')
plt.title('Persons 65 years and over, percent vs. Guns per capita, 2016');


# <a id='conclusions'></a>
# ## Conclusions
# 
# The notes from the GitHub page about the data (https://github.com/BuzzFeedNews/nics-firearm-background-checks/blob/master/README.md) offer the following caveats:
# 
# - An increase in background check requests may signify an increase in gun purchases, but it's not a direct correlation. The laws in each state vary in terms of when checks are required to be submitted and resubmitted and how often. Therefore, while an increase in background checks may indicate more gun purchases in total, it also may indicate stricter regulations (for example, Kentucky runs a check every month for each concealed carry permit holder).
# 
# ### Conclusions from this report
# 
# - Between 2016 and 2010, Kentucky had a 54% increase in background checks for guns. The caveat above would certainly seem to support why Kentucky's permit totals are twice as high as the next state.
# 
# - There appears to be some correlation between population per square mile and guns per capita. The fewer people per square mile the more guns. Intuitively this makes sense as those who live in sparsely populated areas are more likely to need guns to hunt, protect livestock, and possible for personal safety in absense of a more robust law enforcement presence.
# 
# - There does not appear to be any strong correlation between race, education, or age, and guns checks per capita.
# 
# - I don't believe the FBI data is sufficient enough to show much of a correlation with the Census data. While the Census data is more complete, the FBI data has a few issues. First, the data for each state wasn't complete. Only the year-month, state, and grand totals were complete. All the other facts were missing values. Furtermore, the FBI data doesn't take into account the effect of different laws in each state. This report didn't attempt to develop a model to mitigate the distorting effects of different laws on each state's gun background check numbers. To do a fair comparison from state to state, one would need to account for the effects of laws in each state requiring background checks.
# 
# - Also, I did not attempt any true statistical work - only intuitive correlations based on visual inspection. This was a very simple exploration. Further investigation would be required to find stronger correlations using more rigorous statistical analysis.
# 
# ## Submitting your Project 
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[67]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


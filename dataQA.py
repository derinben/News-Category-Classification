import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset and dropping the unused columns for this model
dataset = pd.read_json('News_Category_Dataset_v2.json', lines=True)
dataset.drop(['authors', 'link', 'date'], axis=1, inplace=True)

# Reducing the categories by groupping them
'''
Feel free to edit the group and group names inorder to achieve 
a more balanced dataset and better predictions
'''

categories = dataset['category'].value_counts().index


def groupper(grouplist, name):
    for ele in categories:
        if ele in grouplist:
            dataset.loc[dataset['category'] == ele, 'category'] = name


groupper(grouplist=['WELLNESS', 'HEALTHY LIVING', 'HOME & LIVING', 'STYLE & BEAUTY', 'STYLE'],
         name='LIFESTYLE AND WELLNESS')
groupper(grouplist=['PARENTING', 'PARENTS', 'EDUCATION', 'COLLEGE'], name='PARENTING AND EDUCATION')
groupper(grouplist=['SPORTS', 'ENTERTAINMENT', 'COMEDY', 'WEIRD NEWS', 'ARTS'], name='SPORTS AND ENTERTAINMENT')
groupper(grouplist=['TRAVEL', 'ARTS & CULTURE', 'CULTURE & ARTS', 'FOOD & DRINK', 'TASTE'],
         name='TRAVEL-TOURISM & ART-CULTURE')
groupper(grouplist=['WOMEN', 'QUEER VOICES', 'LATINO VOICES', 'BLACK VOICES'], name='EMPOWERED VOICES')
groupper(grouplist=['BUSINESS', 'MONEY'], name='BUSINESS-MONEY')
groupper(grouplist=['THE WORLDPOST', 'WORLDPOST', 'WORLD NEWS'], name='WORLDNEWS')
groupper(grouplist=['ENVIRONMENT', 'GREEN'], name='ENVIRONMENT')
groupper(grouplist=['TECH', 'SCIENCE'], name='SCIENCE AND TECH')
groupper(grouplist=['FIFTY', 'IMPACT', 'GOOD NEWS', 'CRIME'], name='GENERAL')
groupper(grouplist=['WEDDINGS', 'DIVORCE', 'RELIGION', 'MEDIA'], name='MISC')

# Plot to see if the data is balanced
fig = plt.figure(figsize=(10, 20))
plt.pie(dataset['category'].value_counts().values,
        labels=dataset['category'].value_counts().index,
        autopct='%1.1f%%');

#Dropping the duplicate and empty values

df = dataset.copy() #Let's create a copy of the data frame
df.drop_duplicates(keep='last', inplace=True)
df.drop_duplicates(subset=['short_description','headline'],keep='last',inplace=True) #drops duplicates under 'short_description' and 'headline'

df.dropna(subset=['headline'], inplace=True)
df.dropna(subset=['short_description'], inplace=True)




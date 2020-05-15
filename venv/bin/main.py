import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Get the data

#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')

# Let's see how the data looks
pd.set_option('display.max_columns', 5)
pd.set_option('display.width', 1000)
print(movies_df.head(10))

# Step 2: Data Wrangling
# We need to harvest the year out of the title.
# We can use this general expression to
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#removing any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

print(movies_df.head())

# Looks good.

# We can drop the genres, since we don't need it for this particular system
# Remember, this method is NOT based on the similarity of the content.
# Instead, it's based on the similarity of the users.

movies_df.drop(axis=1, columns="genres", inplace=True)

print(movies_df.head())

# Ratings now
print(ratings_df.head())

# Drop the timestamp

ratings_df.drop(axis=1, columns="timestamp", inplace=True)

print(ratings_df.head())

# Much better

# Step 3: Collaborative Filtering
# Assume we have the following input
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)

# SO we need the movie IDs of the movies inputted

#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)

print(inputMovies.head())

# Now that we have the ratings and the users movies with ratings, we can create
# a matrix with the users and the movies
# Get the user subset that has watched the same movies as the input user

userSubset = ratings_df[ratings_df["movieId"].isin(inputMovies["movieId"].tolist())]
print(userSubset.head())

# We can then group the results by userID
userSubsetGroup = userSubset.groupby(["userId"])
print(userSubsetGroup.get_group(1130))

# This is great, but it would be better if the users most similar to the input
# User are near the top. This will enable us to not go through the entire list and
# Create a richer recommendation while saving some resources.
# index at 0 is the index, 1 is the dataframe

# Note that this is an expensive operation, but it's still better
# to spend O(nlogn) time sorting than it is to use the filtering on the entire dataset.
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)

# Now, lets just use the first 100 users
userSubsetGroup = userSubsetGroup[0:100]

# We will now calculate the similarity
# We will use the pearson similarity

# Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

# For every user group in our subset
inputMovies = inputMovies.sort_values(by='movieId')
for name, group in userSubsetGroup:
    # Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')

    # Get the N for the formula
    nRatings = len(group)
    # Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    # And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    # Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    # Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
    Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
    Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(tempGroupList) / float(
        nRatings)

    # If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
    else:
        pearsonCorrelationDict[name] = 0

# print(pearsonCorrelationDict.items())

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
print(pearsonDF.head())

# Lets get the top 50 users
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
print(topUsersRating.head())

# Now we have a matrix of users, their ratings of movies, and the similarity index

# We will now create the weighted rankings matrix
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']

# We can then sum up all of the weights for the movies
#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
print(tempTopUsersRating.head())

# Lets make the recs
#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

# Now, let's view the top 5 scores and IDs
print(recommendation_df.head())
# Note that the output here isn't ranked - that's likely for the better. The actual numerical values of the
# Suggestions are arbitary.
print(movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(5)['movieId'].tolist())])

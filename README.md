# Machine Learning Portfolio - Connor Galvin

This portfolio contains the code, datasets, and analysis of several projects that I have worked on for school and in my own free time.

## Sentiment Analysis on Amazon reviews

This is my final project for EECE 5644, Machine Learning and Pattern Recognition, which I completed Spring 2018 at Northeastern University. I built a sentiment analysis model that predicts the star value of a product review on Amazon given the text of the review. The dataset, which I found on Kaggle, contains 100,000 written product reviews and their star ratings. 

My model experiments with different machine learning algorithms (Naive Bayes, K-Nearest Neighbors, Linear SVC, and SGD). I also tested different methods for feature extraction and vectorization of the text of each review. 

[Dataset](amazon_reviews.csv)

[Python Code](Sentiment-Model.py)

[Written Report](galvin-setiment-analysis-report.pdf)

[Powerpoint Presentation](galvin-sentiment-analysis-ppt.pptx)

## Classification of points on a normal distribution using a discrimination function

This code 


## Manual implementation of K-means clustering 

This code manually implements the steps that are used in K-means clustering, where K is the number of clusters that you the selected data will be placed into. 
First, the features of a data that we want to base our clustering on are selected. In this case, the features are the length and width of the sepal of a flower. 
Then, k random centroids are selected and placed somewhere on the range of data.  Each point in the dataset is assigned to the centroid that is closest to itself, using Euclidean distance in the feature space.
 The mean of every datapoint assigned in each centroid is calculated. Each of the Centroids are then moved to  these new means. 
The process repeats, with each datapoint being assigned to the nearest centroid. The centroids are moved to the new mean. This process continues until the datapoints are stable and stop moving between centroids. After stability has occurred, the set of points belonging to each centroid are named clusters.  


https://connorgalvin512.github.io/

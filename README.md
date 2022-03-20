# String Correlation matrix

I've recently been involved in building a classification model regarding service tickets in aid of automating some tasks which are currently completed manually.

As part of this, I’m dealing with highly variant data, with frequent text in the titles related to attributes/fields that we want to automatically classify and populate.

Similarly titled tickets often possess the same descriptive fields, meaning that if we are able to generalise and say “tickets titled like ‘Title group A’ have an attribute of X” can automate some of this manual work.

Natural language processing is a main theme governing what we're aiming to achieve here, and common techniques such as using Regular Expressions (RegEx) to try to pull out this data can be effective, but is often unable to deal with variations and new scenarios without manual input.

Such tasks can often require a large amount of painstaking manual input to create custom rules, depending on domain knowledge and familiarity with the dataset in question.

This laborious approach inspired me to consider automating this string manipulation process, allowing faster categorisation of the data and deliver analysis and modelling quicker – the value we’re undoubtedly looking for!

__Applications in the wide world__

This is a rather specific example, but it has its applications when handling user input data where there is bound to be variation in the values typed. This could be the result of different naming conventions, abbreviations, manual error, etc.

One possible example could be a survey on broadband providers, where users are asked to rate their current provider along with other information. This may take the form of the following table:

![](https://github.com/MattPCollins/Classification/blob/main/images/table1.png)


Taking a closer look at the [broadband] column, which we might want to group our analysis by, it gives a quick insight into how users may manually input data "correctly", but makes classification based on these values that little bit harder.
*rename to fake companies???
We could go through these one-by-one on a small scale, rectifying the data, find-and-replace searching for recurring redundant strings (such as “uk”), but when dealing with datasets of hundreds of records and trying to spot user self-defined titles can leave you a bit cross-eyed. Before long we may have a new provider being listed in the data set, for example some new users report they are with “Onestream”, “One Stream” or “OneStream Fibre” and our data cleansing process needs to be repeated.
Thus, we have our use case…

-> example, we can now have a company-by-company comparison of customer satisfaction, max monthly cost, avg time with customer etc to deliver these insights


__Process__

By collecting the string data I wanted to compare and performing some very basic pre-processing techniques (setting to lower case, stripping whitespace and removing any common elements which I knew would appear) was a lightweight starting point.
Creating a unique list of these values, we could then compare each one to every other element in the list, and find the "similarity".
Raw titles data:
['BT', 'Sky', 'Sky UK', 'Sky uk ltd', 'Sky uk limited', 'Virgin', 'Virgin Media', 'Vodaphone', 'vodaphone uk', 'talk talk', 'BT', 'Sky', 'Sky UK', 'Sky uk ltd', 'Sky uk limited', 'Virgin', 'Virgin Media', 'Vodaphone', 'vodaphone uk', 'TalkTalk', 'talk talk', 'BT', 'Sky', 'Sky UK', 'Sky uk limited', 'Virgin', 'Virgin Media', 'Vodaphone', 'vodaphone uk', 'TalkTalk']

Unique list:
['bt', 'talk talk', 'vodaphone uk', 'talktalk', 'virgin media', 'sky uk', 'sky', 'virgin', 'vodaphone']

->Python function, but what does it actually do??? (What is the maths)
We’re using the SequenceMatcher() class in the difflab package in Python to achieve this. This looks to directly compare two strings and find the longest contiguous matching subsequence.
Why choose this over fuzzy matching?
Fuzzy matching (or Approximate String Matching) is used to identify strings that are similar but not exactly the same. Within this, there are several algorithms which can be used, such as cosine similarity and the Levenshtein distance. These are valid approaches for what we’re trying to achieve here and model comparison and efficiency can be tested further down the line.

Using SequenceMatcher() gave me a matrix detailing a quantitative correlation between each of the values. I’ve used the Plotly package to produce the output matrix plot as this interactive and we only need see the data when we’re highlighting the cells in question. 
Note: These plots are very useful for our understanding on smaller samples of data like the one we’re using but quickly become unruly when comparing large datasets. As such it is useful for visualising during our Exploratory Data Analysis, and the plotting can be left out when creating our model. More on these performance considerations later. (chunking data into subsets before final comparisons)

![](https://github.com/MattPCollins/Classification/blob/main/images/matrix.png)


The idea now is to associate any values which have a high enough correlation. I chose an arbitrary value of 0.8 (80%) similarity to start with - this can be tuned further down the line.
This can be an iterative process to determine what correlation we actually need to represent. Ideally we want to choose a threshold which allows a great amount of independence in the output values, without being too strict (at 99% similarity you are preventing “vodaphone” and “vodaphone uk" from being grouped together).
Note: the length of the strings you want to compare will also have weight on the correlation you want to choose. 
Show graph
In the broadband provider example above, we’ve tuned the similarity threshold to 0.5 after evaluating the effectiveness of different thresholds on the data set.

Once we've grouped the similar items, we find the greatest common substring - this will be our generalised value. These generalised values are added to a new list along with anything unique in the original list that was not possible to be grouped.

This reduces our feature set down from 9 elements to the following 5:
['bt', 'sky', 'virgin', 'talk', 'vodaphone']

This is where the domain knowledge comes into play – the data is “good enough” for the classification we’re trying to achieve. We know there is no company called “talk”, but we know it represents Talk Talk exclusively, thus can safely use it as a feature.
Notes:
Domain knowledge and understanding of the data itself is still key - being able to quickly filter out anything at the start will have its benefits for improving the efficiency of the correlation matrix. Finding the right balance of using data cleaning for obvious winners that will help improve efficiency in the correlation matrix itself and using the correlation matrix may help with data cleaning, which may be a manually intensive effort.

e.g. the pseudocode


    if title contains ‘vodaphone’:
	    then title = ‘vodaphone’ 
etc.
Doing this for every provider becomes unscalable very quickly and may be subject to bias when using a sample data set which is not necessarily representative of the complete data.

We can evaluate a suitable correlation threshold by looking at the independence of the values in our result set. If these are deemed to be independent of each other, then we’ve created a result set that can effectively be used to simply out input data. 
Again, it is worth taking consideration on the size of the values we’re comparing as similarity thresholds will be dependent on this.
For example, if we are dealing with short strings (‘sky’,’virgin’) vs longer strings (‘complicated evaluation metric’, ‘simple evaluation metric’), then we might require different thresholds.
We’ll also want to remember that at the higher similarity thresholds then we’re being very specific, so domain knowledge of “what to expect” of our data can be important.

A quick plot against a sample like this gives us some useful info about what value might be worth choosing:

![](https://github.com/MattPCollins/Classification/blob/main/images/line_graph.png)


Could cross-fold validation against several random samples and use the mean value to help ascertain threshold to use.


__Tuning considerations:__

Somewhat bespoke, but remember it is still less work reviewing this, quick is generated at low-cost, than building out yourself.
set too high - not enough generalisation (maybe that is okay, depending on what you want the data to do next!) written before but this might be better wording.


__Putting it all together__

We’re in a good place to use this now.
The quickest example is to assume we’re in a place to use this data as-is. In our example, we could classify on this directly
Our DataFrame:

![](https://github.com/MattPCollins/Classification/blob/main/images/table2.png)


We can look at the average values for satisfaction ratings, years a customer has stayed with the broadband provider and their monthly costs:

![](https://github.com/MattPCollins/Classification/blob/main/images/bar_graph.png)


__Usage:__

This is not something that needs to be run on a daily basis.
depending on the frequency that your data changes and when new cases are input, the generalisations produced in our result set (if stored) can be applied to new data entries. Updating this list of our generalised result sets to be updated at user-defined frequencies away from whatever ML model and analytics solution built alongside.
Performance considerations
	•	Scalability of solution we are comparing all elements of x2 matrix of x elements.
	•	order of magnitude when comparing 10 vs 100 vs 1000, becomes increasingly complex.
	•	demo of this (run times, note visualisations become impossible at this point)
	•	can use partitions of the sample to stop at stable state
	•	collect these results and then iterate through
	•	can run in parallel (RDD if happen to be using Spark, or concurrent.futures, or similar python concurrency tools)

__Closing Thoughts__

where else can be applied
I’ve found a particular use case when looking at high variation in data column
(tsql select distinct title counts)
what value does it bring:
    reduces time in the data cleansing/processing phase
    automating away from heavy manual tasks
    more time spent on the interesting parts of an analytics project!

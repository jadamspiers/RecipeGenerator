import matplotlib.pyplot as plt
import sys
import get_recipes 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
#program to do basic analysis on the data set and report results to the command line or using graphs

# get raw dataset
dataset_raw = get_recipes.load_dataset(silent=True)

#clean/validate dataset
dataset = [recipe for recipe in dataset_raw if get_recipes.clean_recipe(recipe)]


#for stripping out quantity information
cutlist = ["ounces", "ounce", "pounds", "pound", "lbs", "lb", "tsp", "teaspoons", "teaspoon", 
"tbsp", "tablespoons", "tablespoon", "cups", "cup", "cans", "can", "bottles", "bottle", 
"packages","package", "containers", "container", "pinch of" , "pinch"]

ingredientlist = []
ingrednumlist = []
freqdict = {}
df = pd.DataFrame()

#debug paramters
i = 0
pint = 1000
printout = False
debug = False

#pull data from dataset
print("Analyzing Data")
for recipe in dataset:
	#each ingredient is a string 
	recipeingred = []
	for ingredient in recipe['ingredients']:
		#this shows up in the string for some reason, remove it
		ingredient = ingredient.replace("ADVERTISEMENT", '')

		#don't include empty strings
		if len(ingredient) == 0:
			continue
		if debug:
			print(ingredient)
		#process ingredient list
		#rmove cases where a line is talking about a part of the recipe, not an ingredient
		if "For " in ingredient:
			continue

		#remove stuff in parenthesis
		opparen = ingredient.find('(')
		while opparen > 0:
			clparen = ingredient.find(')')
			ingredient = ingredient[:opparen] + ingredient[clparen+1:]
			opparen = ingredient.find('(')
			if clparen < 0:
				#break out if no closing paren
				ingredient = ingredient[:opparen]
				break
			if debug:
				print("clearing parens")


		#remove preparation steps/comments from ingredient 
		commaind = ingredient.find(',')
		if commaind > 0 and ingredient.find("chicken") < 0: #exempt the phrase "boneless, skinless chicken"
			ingredient = ingredient[:commaind]
			if debug:
				print("clearing comma")
		#same case but with - 
		dashind = ingredient.find(' - ')
		if dashind > 0:
			ingredient = ingredient[:dashind]
			if debug: 
				print("clearing dash")

		#remove specific comments that often appear without a comma
		ingredient = ingredient.replace("to taste", '')
		ingredient = ingredient.replace("for garnish", '')
		ingredient = ingredient.replace(' or ', ' ') #remove cases such as "can or bottle"
		ingredient = ingredient.replace('eggs', 'egg') #singularize eggs
		
		#pull ingredients out of cleaned string 
		ingredsplstr = [word for word in ingredient.split() if not word  in cutlist 
		and not	'/' in word #fractions
		and not word.isnumeric() ] #numbers
		
		#coallate ingredient words into one thing
		ingredstr = ' '.join(ingredsplstr).lower()

		#keep track of ingredients in each recipe
		recipeingred.append(ingredstr)

		#build dataframe for correlation mat
		df.loc[i, ingredstr ]= 1
		#keep track of overall ingredient frequency
		if not ingredstr in freqdict.keys():
			freqdict[ingredstr] = 1
		else:
			freqdict[ingredstr] += 1 

	ingredientlist.append(recipeingred)
	ingrednumlist.append(len(recipeingred))
	
	#debug
	if i % pint == 0 and printout:
		print("Iteration: " + str(i))
	i+= 1

#correlation matrix
print("Building Correlation Matrix")
print(df.head())
df = df.fillna(0)
corrmat = df.corr()
shortkeys = list(plotfreqdict.keys())[:5]
print("Correlation matrix:")
print(corrmat.loc[shortkeys,shortkeys])

freqdict = dict(sorted(freqdict.items(), key=lambda x: x[1], reverse=True))

#extract n most common keys for plotting
n = 50
plotfreqdict = {}
for key, value in freqdict.items():
	if len(plotfreqdict) < n:
		plotfreqdict[key] = value
	else:
		break

#kmeans 
#transform ingredient list to usable form
print("Performing K Means Analysis")
ingredientlistvectorizable = [' '.join(x) for x in ingredientlist]
vectorizer = CountVectorizer( vocabulary = freqdict.keys())
X = vectorizer.fit_transform(ingredientlistvectorizable)

#check silhoutte score to pick k
clusters = [2, 3, 4, 5, 6]
silscore = []
for k in clusters:
	kmeans = KMeans(n_clusters = k)
	labels = kmeans.fit_predict(X)
	silscore.append(silhouette_score(X, labels))

#plot silhouetter results
plt.figure()
plt.plot(clusters, silscore)
plt.xlabel("Number of Clusters")
plt.ylabel("Average Silhouette Score")
plt.title("K Means Clustering Analysis")

#freq plot of type of ingredients 
x = range(len(plotfreqdict))
y = list(plotfreqdict.values())
freqtot = sum(y)
for i in range(len(y)):
	y[i] = y[i]/freqtot
plt.figure()
plt.bar(x, y)
plt.xticks(x, plotfreqdict.keys(), rotation='vertical')
plt.ylabel("Frequency")
plt.title("Total Frequency of Ingredients")

#hist of number of ingredients per recipe
plt.figure()
plt.hist(ingrednumlist,  density=True)
plt.xlabel("Number of Ingredients")
plt.ylabel("Frequency")
plt.title("Ingredients Per Recipe")

plt.show()
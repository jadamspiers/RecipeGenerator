from allrecipes import AllRecipes

# Search :
query_options = {
    "wt": "pork curry",
    "ingIncl": "olives",
    "ingExcl": "onions salad",
    "sort": "re"
}
query_result = AllRecipes.Search(query_options)

# Get :
main_recipe_url = query_result[0]['url']
detailed_recipe = AllRecipes.get(main_recipe_url)

print("## %s :" % detailed_recipe['name'])

for ingredient in detailed_recipe['ingredients']:
    print("- %s" % ingredient)

for step in detailed_recipe['steps']:
    print("# %s" % step)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os
import json
import zipfile

# cache the recipes so that we don't have to keep fetching
cache_dir = './tmp'
pathlib.Path(cache_dir).mkdir(exist_ok=True)    # make sure path exists

# download and open dataset
dataset_file_name = 'recipes_raw.zip'
dataset_file_origin = 'https://storage.googleapis.com/recipe-box/recipes_raw.zip'

# use tenserflow's get file function
dataset_file_path = tf.keras.utils.get_file(
    fname=dataset_file_name,
    origin=dataset_file_origin,
    cache_dir=cache_dir,
    extract=True,
    archive_format='zip'
)

# loads the dataset from the file and store json objects in list
def load_dataset(silent=False):
    # name of allrecipes json file
    # note: can add more datasets here
    dataset_file_names = ['recipes_raw_nosource_ar.json']
    
    dataset = []

    # iterate through dataset files
    for dataset_file_name in dataset_file_names:
        dataset_file_path = f'{cache_dir}/datasets/{dataset_file_name}'

        with open(dataset_file_path) as dataset_file:
            json_data_dict = json.load(dataset_file)
            json_data_list = list(json_data_dict.values())
            dict_keys = [key for key in json_data_list[0]]
            dict_keys.sort()
            dataset += json_data_list

    return dataset  


# throw out recipes that are incomplete (don't have one of the required keys)
def clean_recipe(recipe):
    required_keys = ['title', 'ingredients', 'instructions']
    
    if not recipe:
        return False
    
    for required_key in required_keys:
        if not recipe[required_key]:
            return False
        
        if type(recipe[required_key]) == list and len(recipe[required_key]) == 0:
            return False
    
    return True


# convert json recipe object into string
def recipe_to_string(recipe):
    STOP_WORD_TITLE = 'TITLE: '
    STOP_WORD_INGREDIENTS = '\nINGREDIENTS: \n\n'
    STOP_WORD_INSTRUCTIONS = '\nINSTRUCTIONS: \n\n'

    # this shows up at the end of the recipe for some reason, remove it
    noize_string = 'ADVERTISEMENT'
    
    title = recipe['title']
    ingredients = recipe['ingredients']
    instructions = recipe['instructions'].split('\n')
    
    ingredients_string = ''
    for ingredient in ingredients:
        ingredient = ingredient.replace(noize_string, '')
        if ingredient:
            ingredients_string += f'• {ingredient}\n'
    
    instructions_string = ''
    for instruction in instructions:
        instruction = instruction.replace(noize_string, '')
        if instruction:
            instructions_string += f'▪︎ {instruction}\n'
    
    return f'{STOP_WORD_TITLE}{title}\n{STOP_WORD_INGREDIENTS}{ingredients_string}{STOP_WORD_INSTRUCTIONS}{instructions_string}'







################## TEST FUNCTIONS ###################

# get raw dataset
dataset_raw = load_dataset()  
print('Total number of raw examples: ', len(dataset_raw))

# remove invalid recipes
dataset_validated = [recipe for recipe in dataset_raw if clean_recipe(recipe)]
print('Dataset size BEFORE validation', len(dataset_raw))
print('Dataset size AFTER validation', len(dataset_validated))
print('Number of invalid recipes', len(dataset_raw) - len(dataset_validated))

# convert the recipes from json object to string
dataset_stringified = [recipe_to_string(recipe) for recipe in dataset_validated]
print('Stringified dataset size: ', len(dataset_stringified))

for recipe_index, recipe_string in enumerate(dataset_stringified[:10]):
    print('Recipe #{}\n---------'.format(recipe_index + 1))
    print(recipe_string)
    print('\n')

print(dataset_stringified[200]) # example recipe string output
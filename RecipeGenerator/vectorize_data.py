from extract_recipes import *

'''This script vectorizes the list of recipe strings'''

# convert recipe sequence to string
def convert_recipe(recipe_sequence):
    recipeString = vectorize.sequences_to_texts([recipe_sequence])[0]
    recipeString = recipeString.replace('   ', '_').replace(' ', '').replace('_', ' ')


# EOF indicator of recipe
INDICATOR = '␣'

vectorize = tf.keras.preprocessing.text.vectorize(
    char_level=True,
    lower=False,
    filters='',
    split=''
)

vectorize.fit_on_texts([INDICATOR])
vectorize.fit_on_texts(dataset_filtered)
vectorize.get_config()

table_length = len(vectorize.word_counts) + 1

recipe_vectors = vectorize.texts_to_sequences(dataset_filtered)
max_index_example = np.max(recipe_vectors)
convert_recipe(recipe_vectors[0])


# pad the vectors so that all of them are of the same size
padding_no_end = tf.keras.preprocessing.sequence.pad_sequences(
    recipe_vectors,
    maxlen=MAX_RECIPE_LENGTH-1,
    padding='post',
    truncating='post',
    value=vectorize.texts_to_sequences([INDICATOR])[0]
)

padding = tf.keras.preprocessing.sequence.pad_sequences(
    padding_no_end,
    padding='post',
    truncating='post',
    maxlen=MAX_RECIPE_LENGTH+1,
    value=vectorize.texts_to_sequences([INDICATOR])[0]
)

max_index_example = np.max(padding)

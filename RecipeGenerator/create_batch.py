from vectorize_data import *

'''This forms the training batches for the model'''

# form the input and target examples
def form_it_batch(recipe):
    i_seq = recipe[:-1]
    t_seq = recipe[1:]
    
    return i_seq, t_seq

recipes_data = tf.data.Dataset.from_tensor_slices(padding)

for recipe in recipes_data.take(1):
    convert_recipe(recipe.numpy())

recipes_data_targeted = recipes_data.map(form_it_batch)

# vectorize the input and target sequences to text
for input_seq, target_seq in recipes_data_targeted.take(1):
    i_string = vectorize.sequences_to_texts([input_seq.numpy()[:20]])[0]
    t_string = vectorize.sequences_to_texts([target_seq.numpy()[:20]])[0]
    
batches = 50
shuffle_count = 800

recipes_data_train = recipes_data_targeted \
    .shuffle(shuffle_count) \
    .batch(batches, drop_remainder=True) \
    .repeat()

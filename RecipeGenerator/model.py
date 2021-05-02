from organize_data import *

'''This script constructs the model'''

# create and add layers to model
def model(tabLen, embedFrame, rnns, batches):
    layers = tf.keras.models.Sequential()

    layers.add(tf.keras.layers.Embedding(
        output_dim=embedFrame,
        input_dim=tabLen,
        batch_input_shape=[batches, None]
    ))

    layers.add(tf.keras.layers.LSTM(
        units=rnns,
        stateful=True,
        return_sequences=True,
        recurrent_initializer=tf.keras.initializers.GlorotNormal()
    ))

    layers.add(tf.keras.layers.Dense(tabLen))
    
    return layers

# model training details for the embedding layer
seq_len=9
embed=3
inputSize=6
batches=2

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(
  input_dim=seq_len,
  output_dim=embed,
  input_length=inputSize
))

charInput = np.random.randint(
  low=0,
  high=seq_len,
  size=(batches, inputSize)
)
model.compile('rmsprop', 'mse')
tmp_output_array = model.predict(charInput)

# create the final model
tabLen = table_length
embedFrame = 256
rnns = 1024

final_model = model(tabLen, embedFrame, rnns, batches)

final_model.summary()

for inputBat, targetBat in dataset_train.take(1):
    batPred = final_model(inputBat)


logitArr = [[-0.9, 0, 0.9],];

tmp_samples = tf.random.categorical(
    logits=logitArr,
    num_samples = 5
)

charIndexTable = tf.random.categorical(
    num_samples = 1,
    logits=batPred[0]
)

charIndexTable.shape

charIndexTable = tf.squeeze(
    input=charIndexTable,
    axis=-1
).numpy()

charIndexTable.shape

charIndexTable[:100]
# test the final model with custom batches
for inputBat_custom, targetBat_custom in dataset_train.take(1):
    random_input = np.zeros(shape=(batches, 10))
    batPred_custom = final_model(random_input)

from model import *

'''This script trains our final model'''

def loss(labels, logits):
    entropy = tf.keras.losses.sparse_categorical_crossentropy(
      y_true=labels,
      y_pred=logits,
      from_logits=True
    )
    
    return entropy

loss = loss(target_example_batch, example_batch_predictions)

optimizeFunc = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizeFunc,
    loss=loss
)

checkpoint_dir = 'tmp/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)


callbackStop = tf.keras.callbacks.EarlyStopping(
    patience=5,
    monitor='loss',
    restore_best_weights=True,
    verbose=1
)

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

firstEpoch  = 0
epoch_increment = 1
epochs = firstEpoch + epoch_increment
epochSteps = 2

computed_epochs = {}

computed_epochs[firstEpoch] = model.fit(
    x=dataset_train,
    epochs=epochs,
    steps_per_epoch=epochSteps,
    initial_epoch=firstEpoch,
    callbacks=[
        checkpoint_callback,
        callbackStop
    ]
)

modelName = 'receip_generation' + str(firstEpoch) + '.h5'
model.save(modelName, save_format='h5')


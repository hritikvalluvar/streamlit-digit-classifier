import tensorflow as tf

def train_mnist():

    # Setting callback for 99% accuracy
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
          print("\nReached 99% accuracy so cancelling training!")
          self.model.stop_training = True

    # Providing Data
    mnist = tf.keras.datasets.mnist

    # x_train -> training_images, y_train -> training_labels, x_test -> test_images, y_test -> test_labels
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    # Normalizing
    x_train  = x_train / 255.0
    x_test = x_test / 255.0

    callbacks = myCallback()

    # Creating Neural Network which has 2 layers
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = tf.nn.relu), 
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])

    # Compiling our Neural Network
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Model Fitting
    history = model.fit(
        x_train,
        y_train,
        epochs = 10,
        callbacks=[callbacks]
    )

    # Evaluating our model
    results = model.evaluate(x_test, y_test)

    print("\n\nTest accuracy: ", results[1])

    #saving model
    model.save('mnist_model')


    return history.epoch, history.history['accuracy'][-1]

train_mnist()
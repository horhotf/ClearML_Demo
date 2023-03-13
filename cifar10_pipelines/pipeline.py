from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes
import tensorflow as tf

@PipelineDecorator.component(return_values=['X_train', 'X_test', 'y_train', 'y_test'], cache=True, task_type=TaskTypes.data_processing)
def step_one():
    import tensorflow as tf

    cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    return X_train, X_test, y_train, y_test

@PipelineDecorator.component(
  return_values=['X_train'], cache=True, task_type=TaskTypes.data_processing)
def step_two_1(X_train):
    import tensorflow as tf
    
    return tf.keras.utils.normalize(X_train, axis=1)

@PipelineDecorator.component(
  return_values=['X_test'], cache=True, task_type=TaskTypes.data_processing)
def step_two_2(X_test):
    import tensorflow as tf
    
    return tf.keras.utils.normalize(X_test, axis=1)


@PipelineDecorator.component(return_values=['accuracy'], cache=True, task_type=TaskTypes.training)
def step_three(X_train, y_train, X_test, Y_test):
    print('step_three')
    import tensorflow as tf
    from keras.regularizers import l2
    from keras.optimizers import SGD
    # make sure we have pandas for this step, we need it to use the data_frame

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

    model.fit(X_train, y_train, batch_size=64, shuffle=True, epochs=3)

    val_loss, val_acc = model.evaluate(X_test, Y_test)

    return val_acc

@PipelineDecorator.component(return_values=['accuracy'], cache=True, task_type=TaskTypes.qc)
def step_four(model, X_data, Y_data):

    val_loss, val_acc = model.evaluate(X_data, Y_data)

    return val_acc

@PipelineDecorator.pipeline(name='pipeline', project='examples', version='0.1')
def main():
    X_train, X_test, y_train, y_test  = step_one()
    X_train = step_two_1(X_train)
    X_test = step_two_2(X_test)
    score = step_three(X_train, y_train, X_test, y_test)
    accuracy = 100 * score
    print(f"Accuracy={accuracy}%")

if __name__ == '__main__':
    # run the pipeline on the current machine, for local debugging
    # for scale-out, comment-out the following line and spin clearml agents
    PipelineDecorator.run_locally()

    main()
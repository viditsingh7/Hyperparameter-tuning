import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# App title
st.title("🧠 EMNIST Hyperparameter Tuning")
st.markdown("Train a digit recognizer on EMNIST with customizable hyperparameters.")

# Sidebar sliders
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
epochs = st.sidebar.slider("Epochs", 1, 10, 5)
batch_size = st.sidebar.slider("Batch Size", 32, 256, 64, step=32)
optimizer_name = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])

# Load and preprocess EMNIST
def load_data():
    (train, test), _ = tfds.load(
        'emnist/digits',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )
    return train, test

def preprocess(dataset, batch_size):
    return dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)) \
                  .shuffle(1000) \
                  .batch(batch_size) \
                  .prefetch(tf.data.AUTOTUNE)

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def train_model(lr, optimizer_name, epochs, batch_size):
    train_data = preprocess(ds_train, batch_size)
    test_data = preprocess(ds_test, batch_size)

    optimizers = {
        "adam": tf.keras.optimizers.Adam(lr),
        "sgd": tf.keras.optimizers.SGD(lr),
        "rmsprop": tf.keras.optimizers.RMSprop(lr)
    }
    optimizer = optimizers.get(optimizer_name, tf.keras.optimizers.Adam(lr))

    model = create_model()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data, validation_data=test_data, epochs=epochs, verbose=1)
    return history

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(history.history['accuracy'], label='Train Acc')
    ax[1].plot(history.history['val_accuracy'], label='Val Acc')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    st.pyplot(fig)

# Load data
with st.spinner("Loading EMNIST..."):
    ds_train, ds_test = load_data()

# Train button
if st.button("🚀 Train Model"):
    st.write(f"Training with: `lr={learning_rate}`, `epochs={epochs}`, `batch={batch_size}`, `optimizer={optimizer_name}`")
    with st.spinner("Training in progress..."):
        history = train_model(learning_rate, optimizer_name, epochs, batch_size)
        st.success("Training complete!")
        plot_history(history)

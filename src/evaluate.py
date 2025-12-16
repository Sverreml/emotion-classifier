import matplotlib.pyplot as plt
import keras

def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.plot(acc, label="train acc")
    plt.plot(val_acc, label="val acc")
    plt.legend()
    plt.show()

    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.legend()
    plt.show()


def evaluate(model_path, test_ds):
    model = keras.models.load_model(model_path)
    return model.evaluate(test_ds)

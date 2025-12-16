from datasets import load_datasets
from train import train_model
from evaluate import plot_history, evaluate
from config import MODEL_DIR, VERSION

train_ds, val_ds, test_ds = load_datasets()
model, history = train_model(train_ds, val_ds)

plot_history(history)

test_loss, test_acc = evaluate(MODEL_DIR / "best_model.keras", test_ds)
print(f"Test accuracy: {test_acc:.3f}")

model.save(MODEL_DIR / f"final_classifier_{VERSION}.keras")

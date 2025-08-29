from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_iris():
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
    df = df.rename(columns={"target": "species"})
    return df

def train_and_save(out_path: Path, epochs: int = 50, batch_size: int = 16) -> float:
    df = load_iris()
    X = df.drop(columns=["species"]).values.astype("float32")
    y = tf.keras.utils.to_categorical(df["species"].values, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)

    print(f"Saved TensorFlow model to {out_path} with accuracy={acc:.3f}")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Train a TensorFlow model on Iris dataset.")
    parser.add_argument("--out", type=Path, default=Path("models/tf_model"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_and_save(args.out, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description="Make a prediction with a trained TF model.")
    parser.add_argument("--model", type=Path, default=Path("models/tf_model"))
    parser.add_argument("--sepal_length", type=float, required=True)
    parser.add_argument("--sepal_width", type=float, required=True)
    parser.add_argument("--petal_length", type=float, required=True)
    parser.add_argument("--petal_width", type=float, required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    X = np.array([[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]])
    pred = np.argmax(model.predict(X), axis=1)[0]
    print(f"Predicted class: {pred}")

if __name__ == "__main__":
    main()
tensorflow
pandas
numpy
matplotlib
jupyter
ipykernel
black
flake8
pytest
pre-commit


python src/train.py --out models/tf_model --epochs 50

python src/predict.py --model models/tf_model \
  --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2

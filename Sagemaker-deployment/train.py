import argparse, os
import boto3
import sagemaker

import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()

    # Can have other hyper-params such as batch-size, which we are not defining in this case
    parser.add_argument('--epochs',
                        type=int,
                        default=10)
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001)

    # sm_model_dir: model artifacts stored here after training
    # training directory has the data for the model
    parser.add_argument('--sm-model-dir',
                        type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir',
                        type=str)
    parser.add_argument('--train',
                        type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'))

    args, _ = parser.parse_known_args()
    epochs = args.epochs
    lr = args.learning_rate
    model_dir = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir = args.train

    train_df = pd.read_csv(training_dir + '/train.csv', sep=',')

    X, y = train_df.drop('loan_status', axis=1), train_df.loan_status
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    def nn_model(
        num_columns: int,
        num_labels: int,
        hidden_units: list,
        dropout_rates: list,
        learning_rate: float,
    ):
        inp = tf.keras.layers.Input(shape=(num_columns, ))
        x = BatchNormalization()(inp)
        x = Dropout(dropout_rates[0])(x)
        for i in range(len(hidden_units)):
            x = Dense(hidden_units[i], activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rates[i + 1])(x)
        x = Dense(num_labels, activation='sigmoid')(x)

        model = Model(inputs=inp, outputs=x)
        model.compile(
            optimizer=Adam(learning_rate),
            loss='binary_crossentropy',
            metrics=[AUC(name='AUC')]
        )
        return model

    num_columns = X_train.shape[1]
    num_labels = 1
    hidden_units = [150, 150, 150]
    dropout_rates = [0.1, 0, 0.1, 0]

    model = nn_model(
        num_columns=num_columns,
        num_labels=num_labels,
        hidden_units=hidden_units,
        dropout_rates=dropout_rates,
        learning_rate=lr
    )

    early_stop = EarlyStopping(patience=10)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=[early_stop]
    )

    model.save(os.path.join(sm_model_dir, '000000001'), 'loan_delin_model.h5')
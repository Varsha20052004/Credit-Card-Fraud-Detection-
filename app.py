from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape, BatchNormalization, Dropout
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and preprocess the data (modify the path as needed)
data = pd.read_csv(r"C:\Users\VARSHA\venv\creditcard.csv\ccfd\creditcard_2023.csv")

# Check for missing values and drop if any
if data.isnull().sum().sum() > 0:
    data = data.dropna()

# Standardize the 'Amount' column
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Prepare the dataset for training
X = data.drop(columns=['Class'])
y = data['Class']

# Reshape data for Conv1D
X_reshaped = X.values.reshape(X.shape[0], X.shape[1], 1)
num_features = X.shape[1]
input_shape = (num_features, 1)

import tensorflow as tf
from tensorflow.keras.layers import Layer

class SliceLayer(Layer):
    def call(self, inputs):
        return inputs[:, :30, :]
input_shape = (30, 1)  # Confirmed input shape


# Define the autoencoder model
input_layer = Input(shape=input_shape)

# Encoder
encoder = Conv1D(32, 2, activation='relu', padding='same')(input_layer)
encoder = BatchNormalization()(encoder)
encoder = MaxPooling1D(2, padding='same')(encoder)  # Reduces by half: (29 -> 15)
encoder = Conv1D(16, 2, activation='relu', padding='same')(encoder)
encoder = BatchNormalization()(encoder)
encoder = MaxPooling1D(2, padding='same')(encoder)  # Reduces by half: (15 -> 8)
encoder = Conv1D(8, 2, activation='relu', padding='same')(encoder)
encoder = BatchNormalization()(encoder)
encoder_output_shape = K.int_shape(encoder)

# Flatten and bottleneck
flatten = Flatten()(encoder)
encoded = Dense(16, activation='relu')(flatten)  # Bottleneck layer
encoded = Dropout(0.2)(encoded)

# Decoder
decoder = Dense(np.prod(encoder_output_shape[1:]), activation='relu')(encoded)
decoder = Reshape((encoder_output_shape[1], encoder_output_shape[2]))(decoder)
decoder = Conv1D(8, 2, activation='relu', padding='same')(decoder)
decoder = BatchNormalization()(decoder)
decoder = UpSampling1D(2)(decoder)  # Upsampling: (8 -> 16)
decoder = Conv1D(16, 2, activation='relu', padding='same')(decoder)
decoder = BatchNormalization()(decoder)
decoder = UpSampling1D(2)(decoder)  # Upsampling: (16 -> 32)
decoder = Conv1D(32, 2, activation='relu', padding='same')(decoder)
decoder = BatchNormalization()(decoder)
decoder = Conv1D(1, 2, activation='sigmoid', padding='same')(decoder)

# Use custom slicing layer
decoder = SliceLayer()(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Train on non-fraudulent data only
X_train_non_fraud = X_train[y_train == 0]

autoencoder.fit(X_train_non_fraud, X_train_non_fraud,
                epochs=10,
                batch_size=64,
                validation_data=(X_test, X_test),
                shuffle=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [float(request.form['V' + str(i)]) for i in range(1, 29)]
        amount = float(request.form['Amount'])

        # Create feature array and standardize the 'Amount' feature
        features.append(amount)
        features = np.array(features).reshape(1, -1)
        features[:, -1] = scaler.transform(features[:, -1].reshape(-1, 1)).flatten()

        # Reshape for Conv1D input
        features = features.reshape(1, num_features, 1)

        # Predict using the autoencoder
        reconstructed = autoencoder.predict(features)
        reconstruction_error = np.mean(np.power(features - reconstructed, 2), axis=1)

        # Define threshold (use the threshold determined during training)
        threshold = 0.01  # Update this with your threshold

        # Determine if the transaction is fraudulent
        is_fraud = reconstruction_error > threshold

        result = 'Fraudulent' if is_fraud else 'Non-Fraudulent'
    except Exception as e:
        result = f"Error occurred: {str(e)}"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

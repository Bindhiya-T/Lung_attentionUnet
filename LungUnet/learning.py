import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the model without the optimizer to avoid the error
model = load_model('model.h5', compile=False)

# Modify the optimizer's learning rate manually and recompile the model
optimizer = Adam(learning_rate=1e-4)  # Adjust learning rate as needed
model.compile(optimizer=optimizer, loss='binary_crossentropy')

# Save the model with the updated configuration in .keras format
model.save('updated_model.keras')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the first CNN model
model_1 = Sequential()
model_1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Conv2D(64, (3, 3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Flatten())

# Define the second CNN model
model_2 = Sequential()
model_2.add(Conv2D(32, (3, 3), activation='relu', input_shape=model_1.output_shape[1:]))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Conv2D(64, (3, 3), activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Flatten())

# Combine the two models
combined_model = Sequential()
combined_model.add(model_1)
combined_model.add(model_2)
combined_model.add(Dense(128, activation='relu'))
combined_model.add(Dense(num_classes, activation='softmax'))

# Compile and train the model
combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
combined_model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

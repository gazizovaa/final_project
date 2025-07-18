import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import streamlit as st 
import tensorflow as tf
import random 
import io 
import cv2 
from helper_functions import walk_through_dir, plot_loss_curves, compare_historys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from PIL import Image 
import torch 
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import pickle
import shutil


# Page Configuration
st.set_page_config(
    page_title="Final Project App",
    layout="wide",
    initial_sidebar_state="auto"
)

# Model Caching
@st.cache_data
def load_trained_model():
    try:
        if os.path.exists('saved_models/best_model.h5'):
            return tf.keras.models.load_model('saved_models/best_model.h5')
        return None 
    except Exception as e:
        st.error(f"Error while loading the model: {e}")
        return None 

@st.cache_data
def load_multiple_models():
    models = {}
    model_paths = {
        'model_1': 'saved_models/cnn_model.h5',
        'model_2': 'saved_models/inception_model.h5',
        'model_3': 'saved_models/mobilenet_model.h5',
        'model_4': 'saved_models/efficientnet_model.h5'
    }

    for model_name, path in model_paths.items():
        if os.path.exists(path):
            models[model_name] = tf.keras.models.load_model(path)
            st.success(f"{model_name} loaded successfully!")
        else:
            st.info(f"{model_name} is not found at {path}.")
    return models 

@st.cache_data
def load_training_history(model_name):
    history_path = f'training_history/{model_name}_history.pkl'  
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            return pickle.load(f)
    return None 

def save_model(model, model_name):
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/{model_name}.h5'
    model.save(model_path)
    st.success(f"Model saved to {model_path}")

def save_training_history(history, model_name):
    os.makedirs('training_history', exist_ok=True)
    history_path = f'training_history/{model_name}_history.pkl'  
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f) 
    st.success(f"Training history saved to {history_path}.")

# Session State Management
def initialize_session_state():
    if 'models_loaded' not in st.session_state:  
        st.session_state.models_loaded = False
        st.session_state.models = {}
        st.session_state.training_complete = False
        st.session_state.data_loaded = False 

initialize_session_state()

# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomHeight(0.2),
    tf.keras.layers.RandomWidth(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

# Download dataset function
def download_dataset():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        if not os.path.exists("data"):
            os.makedirs("data")

        if not os.path.exists("data/Drug Vision/Data Combined"):
            api = KaggleApi()
            api.set_config_value('username', st.secrets["kaggle"]["username"])
            api.set_config_value("key", st.secrets["kaggle"]["key"])
            api.authenticate()
            api.dataset_download_files("vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images", 
                                     path="data", unzip=True)
    except Exception as e:
        st.warning(f"Could not download dataset: {e}")

# Try to download dataset
download_dataset()

# Create train/test directories
def create_train_and_test_directories():
    train_dir = 'train'
    test_dir = 'test'

    # Delete the existing folders beforehand
    shutil.rmtree(train_dir, ignore_errors=True) 
    shutil.rmtree(test_dir, ignore_errors=True)

    # Create new directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True) 

    dir_path = "data/Drug Vision/Data Combined" 
    
    # Create a progress bar
    progress_bar = st.progress(0)
    total_files = len(train_data.filenames) + len(test_data.filenames)
    processed = 0 

    # Copy files to train directory
    for i, file in enumerate(train_data.filenames):
        parts = file.replace('\\', '/').split('/')
        if len(parts) >= 2:
            class_name, file_name = parts[-2], parts[-1] 
            src = os.path.join(dir_path, class_name, file_name)
            train_target_dir = os.path.join(train_dir, class_name)
            os.makedirs(train_target_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(train_target_dir, file_name))

        processed += 1 
        progress_bar.progress(processed / total_files)

    # Copy files to test directory
    for i, file in enumerate(test_data.filenames):
        parts = file.replace('\\', '/').split('/')
        if len(parts) >= 2:
            class_name, file_name = parts[-2], parts[-1]  
            src = os.path.join(dir_path, class_name, file_name)
            test_target_dir = os.path.join(test_dir, class_name)
            os.makedirs(test_target_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(test_target_dir, file_name)) 
        
        processed += 1 
        progress_bar.progress(processed / total_files) 

# Model creation function
def get_or_create_models():
    # Create the models if they are not loaded
    if not st.session_state.models:
        # Model 1 - CNN
        model_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', input_shape=(224,224,3), activation='relu'),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Model 2 - Inception
        input_shape = (224, 224, 3)
        base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                       input_shape=input_shape,
                                                       weights='imagenet')
        base_model.trainable = False
        inputs = tf.keras.layers.Input(shape=input_shape, name='input_layer')
        x = data_augmentation(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax', name='output_layer')(x)
        model_2 = tf.keras.Model(inputs, outputs)

        # Model 3 - MobileNet (Fixed - was using InceptionV3)
        base_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                       input_shape=input_shape,
                                                       weights='imagenet')
        base_model.trainable = False
        inputs = tf.keras.layers.Input(shape=input_shape, name='input_layer')
        x = data_augmentation(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax', name='output_layer')(x)
        model_3 = tf.keras.Model(inputs, outputs)

        # Model 4 - EfficientNet
        base_model = tf.keras.applications.EfficientNetV2B0(include_top=False,
                                                       input_shape=input_shape,
                                                       weights='imagenet')
        base_model.trainable = False
        inputs = tf.keras.layers.Input(shape=input_shape, name='input_layer')
        x = data_augmentation(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax', name='output_layer')(x)
        model_4 = tf.keras.Model(inputs, outputs)

        # Add models to session state
        st.session_state.models = {
            'model_1': model_1,
            'model_2': model_2,
            'model_3': model_3,
            'model_4': model_4 
        }
    return st.session_state.models 

# Main app
st.title("Pharmaceutical Drugs and Vitamins Synthetic Images App")

# Initialize data loading
@st.cache_resource 
def load_data():
    dir_path = "data/Drug Vision/Data Combined"
    if not os.path.exists(dir_path):
        st.error("Dataset path not found!")
        return None, None 
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    tf.random.set_seed(42)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = train_datagen.flow_from_directory(directory=dir_path,
                                                    target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    subset='training')

    test_data = test_datagen.flow_from_directory(directory=dir_path,
                                                    target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    subset='validation')
    return train_data, test_data

# Load data
train_data, test_data = load_data()

# Sidebar
choice = st.sidebar.radio("Selections", ["Dataset Overview", "Fitting the Model", "Preprocessing", 
                                         "Data Augmentation", "Transfer Learning", "Fine Tuning", 
                                         "VLM (Vision Language Model)", "Drugs and Vitamins Detection",
                                         "Performance Evaluation Metrics"])

if choice == "Dataset Overview":
    st.subheader("Dataset Overview")
    
    if os.path.exists("data/Drug Vision"):
        st.text(walk_through_dir("data/Drug Vision"))
        
        if os.path.exists("data/Drug Vision/Data Combined"):
            num_of_classes = len(os.listdir("data/Drug Vision/Data Combined"))
            st.markdown(f"The number of classes: **{num_of_classes}**")
            
            def view_random_image(target_dir, target_classes):
                target_class = random.choice(target_classes)
                target_folder = os.path.join(target_dir, target_class)
                if os.path.exists(target_folder):
                    images = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        random_image = random.choice(images)
                        img = mpimg.imread(os.path.join(target_folder, random_image))
                        fig, ax = plt.subplots()
                        ax.imshow(img) 
                        ax.set_title(target_class)
                        ax.axis("off")
                        st.write(f'Image shape is {img.shape}')
                        st.pyplot(fig)
            
            target_dir = 'data/Drug Vision/Data Combined'
            if os.path.exists(target_dir):
                view_random_image(target_dir=target_dir, target_classes=os.listdir(target_dir))
    else:
        st.error("Dataset not found. Please ensure the dataset is downloaded.")

elif choice == "Fitting the Model":
    st.subheader("Fitting the Model")
    
    # Add GPU checking
    if tf.config.list_physical_devices('GPU'):
        st.success("GPU is available!")
    else:
        st.warning("No GPU found!")
        
    models = get_or_create_models()
    # Create separate tabs for models
    model_tabs = st.tabs([f"Model {i + 1}" for i in range(len(models))])
    
    if st.button("Train all models"):
        with st.spinner("Training Models..."):
            # Add progress bar
            progress_bar = st.progress(0) 
            status_text = st.empty()
            
            for i, (model_name, model) in enumerate(models.items()):
                status_text.text(f"Training {model_name}...")
                
                # Compile models
                model.compile(loss='categorical_crossentropy',
                              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                              metrics=['accuracy'])
                
                # Callbacks
                checkpoint_path = f'checkpoints/{model_name}_checkpoint.weights.h5'
                model_ckp = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    save_best_only=True,
                    save_weights_only=True,
                    save_freq='epoch',
                    monitor='val_accuracy',
                    mode='max'
                )
                
                model_es = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ) 
                
                # Train
                history = model.fit(
                    train_data,
                    epochs=5,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data),
                    callbacks=[model_ckp, model_es],
                    verbose=0
                )
                
                # Save model and history
                save_model(model, model_name)
                save_training_history(history, model_name)
                
                # Store in session state
                st.session_state[f"{model_name}_history"] = history
                save_model(model, model_name) 
                save_training_history(history, model_name) 
                
                progress_bar.progress((i + 1) / len(models))
            
            st.session_state.training_complete = True
            st.success("All models trained successfully!")
            st.balloons() 
    
    # Display all model summary info
    for (model_name, model), tab in zip(models.items(), model_tabs):
        with tab:
            # Model summary info 
            with st.expander("Model Architecture"):
                stream = io.StringIO()
                model.summary(print_fn=lambda x: stream.write(x + '\n'))
                st.text(stream.getvalue())
                
                # Layer table
                layers = [(layer.name, str(layer.output.shape), layer.count_params()) 
                         for layer in model.layers]
                st.table(pd.DataFrame(layers, 
                          columns=["Layer", "Output Shape", "Parameters"]))
            
            # Training results
            if st.session_state.training_complete and f"{model_name}_history" in st.session_state:
                history = st.session_state[f"{model_name}_history"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Accuracy", f"{history.history['accuracy'][-1]:.2%}")
                    st.metric("Training Loss", f"{history.history['loss'][-1]:.4f}")
                with col2:
                    st.metric("Validation Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")
                    st.metric("Validation Loss", f"{history.history['val_loss'][-1]:.4f}")
                
                fig = plot_loss_curves(history)
                st.pyplot(fig)
                
                # Test results
                if st.button(f"Evaluate {model_name}"):
                    loss, accuracy = model.evaluate(test_data, verbose=0)
                    st.success(f"{model_name} Test Accuracy: {accuracy:.2%}")
            else:
                st.warning("Model is not trained yet!")

elif choice == "Preprocessing":
    st.subheader("Preprocessing")
    data_dir = "data/Drug Vision/Data Combined"
    
    if os.path.exists(data_dir):
        class_counts = {}
        total_images = 0

        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_name] = count
                total_images += count

        # Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        classes = list(class_counts.keys())  
        counts = list(class_counts.values())

        ax.bar(classes, counts)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Images')
        ax.set_title('Distribution of Images per Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write(f"**Total Images:** {total_images}")
        st.write(f"**Number of Classes:** {len(classes)}")
    else:
        st.error("Dataset not found.")

elif choice == "Data Augmentation":
    st.subheader("Data Augmentation")
    
    sample_image_path = "data/Drug Vision/Data Combined"
    if os.path.exists(sample_image_path):
        class_folders = os.listdir(sample_image_path)
        selected_class = st.selectbox("Select a class:", class_folders)
        class_path = os.path.join(sample_image_path, selected_class)

        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                sample_img_path = os.path.join(class_path, images[0])
                original_img = tf.keras.utils.load_img(sample_img_path, target_size=(224, 224))
                img_arr = tf.keras.utils.img_to_array(original_img)
                img_arr = tf.expand_dims(img_arr, 0)

                fig, axes = plt.subplots(2, 4, figsize=(15, 8))
                axes = axes.flatten() 

                # Original image
                axes[0].imshow(original_img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                # Augmented images
                for i in range(1, 8):
                    augmented_img = data_augmentation(img_arr)
                    axes[i].imshow(tf.keras.utils.array_to_img(augmented_img[0]))
                    axes[i].set_title(f"Augmented {i}")
                    axes[i].axis("off")

                st.pyplot(fig)
    else:
        st.error("Dataset not found.")

elif choice == "Transfer Learning":
    st.subheader("Transfer Learning")
    st.write("This section shows the performance of different pre-trained models:")
    st.write("- InceptionV3")
    st.write("- MobileNetV2") 
    st.write("- EfficientNetV2B0")
    
    if st.session_state.training_complete:
        st.success("Models have been trained! Check the training histories in session state.")
    else:
        st.info("Train the models first in the 'Fitting the Model' section.")

elif choice == "Fine Tuning":
    st.subheader("Fine Tuning")
    if 'model_2' not in st.session_state.models:
        st.error("Model 2 (InceptionV3) not found. Please train it first.")
        st.stop()
    
    if 'model_2_history' not in st.session_state:
        st.error("Model 2 training history not found. Please train it first.")
        st.stop() 
    
    get_model_2 = st.session_state.models['model_2']
    initial_history = st.session_state['model_2_history']

    st.write("Before Fine-Tuning:")
    get_model_2_layers = get_model_2.layers

    for layer_number, layer in enumerate(get_model_2_layers):
        st.write(f"Layer number: {layer_number} | Layer name: {layer.name} | Trainable?: {layer.trainable}")
    get_model_2_base_model = get_model_2_layers[2] 

    st.write(get_model_2_base_model.name) 
    get_model_2_base_model.trainable = False

    # How many layers are trainable in our model_2_base_model
    st.write(len(get_model_2_base_model.trainable_variables))

    # Check which layers are tunable (trainable)
    for layer_number, layer in enumerate(get_model_2_layers):
        st.write(layer_number, layer.name, layer.trainable)

    # Make all the layers in model_2_base_model trainable
    get_model_2_base_model.trainable = True
    
    # Freeze all layers except for the last 10
    for layer in get_model_2_base_model.layers[:-10]:
        layer.trainable = False
    
    get_model_2.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        metrics=['accuracy'])
    
    for layer_number, layer in enumerate(get_model_2_base_model.layers):
        print(layer_number, layer.name, layer.trainable)
    
    st.write(len(get_model_2_layers))
    
    # Fine tune for another 5 epochs
    initial_epochs = len(initial_history.history['loss'])
    fine_tune_epochs = initial_epochs + 5

    with st.spinner("Fine-tune the InceptionV3 model"):
        # Refit the model
        history_fine_data_aug_2 = get_model_2.fit(train_data,
                                                  epochs=fine_tune_epochs,
                                                  validation_data=test_data,
                                                  initial_epoch=initial_epochs,
                                                  validation_steps=int(0.25 * len(test_data)))
     # Save the fine-tuned history
    st.session_state['model_2_history_fine'] = history_fine_data_aug_2
    
    # Compare histories
    st.subheader("Training History Comparison")
    try:
        fig = compare_historys(
            original_history=initial_history,
            new_history=history_fine_data_aug_2,
            initial_epochs=initial_epochs
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not plot history comparison: {e}")

    st.write("------------------------------------------------------------------")
    if 'model_3' not in st.session_state.models:
        st.error("Model 3 (MobileNet) not found. Please train it first.")
        st.stop()
    
    if 'model_3_history' not in st.session_state:
        st.error("Model 3 training history not found. Please train it first.")
        st.stop() 
    
    get_model_3 = st.session_state.models['model_3']
    initial_history = st.session_state['model_3_history']

    st.write("Before Fine-Tuning:")
    get_model_3_layers = get_model_3.layers

    for layer_number, layer in enumerate(get_model_3_layers):
        st.write(f"Layer number: {layer_number} | Layer name: {layer.name} | Trainable?: {layer.trainable}")
    get_model_3_base_model = get_model_3_layers[2] 

    st.write(get_model_3_base_model.name) 
    get_model_3_base_model.trainable = False

    # How many layers are trainable in our model_3_base_model
    st.write(len(get_model_3_base_model.trainable_variables))

    # Check which layers are tunable (trainable)
    for layer_number, layer in enumerate(get_model_3_layers):
        st.write(layer_number, layer.name, layer.trainable)

    # Make all the layers in model_3_base_model trainable
    get_model_3_base_model.trainable = True
    
    # Freeze all layers except for the last 10
    for layer in get_model_3_base_model.layers[:-10]:
        layer.trainable = False
    
    get_model_3.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        metrics=['accuracy'])
    
    for layer_number, layer in enumerate(get_model_3_base_model.layers):
        print(layer_number, layer.name, layer.trainable)
    
    st.write(len(get_model_3_layers.layers))
    
    # Fine tune for another 5 epochs
    initial_epochs = len(initial_history.history['loss'])
    fine_tune_epochs = initial_epochs + 5

    with st.spinner("Fine-tune the MobileNet model"):
        # Refit the model
        history_fine_data_aug_3 = get_model_3.fit(train_data,
                                                  epochs=fine_tune_epochs,
                                                  validation_data=test_data,
                                                  initial_epoch=initial_epochs,
                                                  validation_steps=int(0.25 * len(test_data)))
     # Save the fine-tuned history
    st.session_state['model_3_history_fine'] = history_fine_data_aug_3
    
    # Compare histories
    st.subheader("Training History Comparison")
    try:
        fig = compare_historys(
            original_history=initial_history,
            new_history=history_fine_data_aug_3,
            initial_epochs=initial_epochs
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not plot history comparison: {e}")

    st.write("------------------------------------------------------------------")
    if 'model_4' not in st.session_state.models:
        st.error("Model 4 (EfficientNet) not found. Please train it first.")
        st.stop()
    
    if 'model_4_history' not in st.session_state:
        st.error("Model 4 training history not found. Please train it first.")
        st.stop() 
    
    get_model_4 = st.session_state.models['model_4']
    initial_history = st.session_state['model_4_history']

    st.write("Before Fine-Tuning:")
    get_model_4_layers = get_model_4.layers

    for layer_number, layer in enumerate(get_model_4_layers):
        st.write(f"Layer number: {layer_number} | Layer name: {layer.name} | Trainable?: {layer.trainable}")
    get_model_4_base_model = get_model_4_layers[2] 

    st.write(get_model_4_base_model.name) 
    get_model_4_base_model.trainable = False

    # How many layers are trainable in our model_4_base_model
    st.write(len(get_model_4_base_model.trainable_variables))

    # Check which layers are tunable (trainable)
    for layer_number, layer in enumerate(get_model_4_layers):
        st.write(layer_number, layer.name, layer.trainable)

    # Make all the layers in model_4_base_model trainable
    get_model_4_base_model.trainable = True
    
    # Freeze all layers except for the last 10
    for layer in get_model_4_base_model.layers[:-10]:
        layer.trainable = False
    
    get_model_4.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        metrics=['accuracy'])
    
    for layer_number, layer in enumerate(get_model_4_base_model.layers):
        print(layer_number, layer.name, layer.trainable)
    
    st.write(len(get_model_4_layers.layers))
    
    # Fine tune for another 5 epochs
    initial_epochs = len(initial_history.history['loss'])
    fine_tune_epochs = initial_epochs + 5

    with st.spinner("Fine-tune the MobileNet model"):
        # Refit the model
        history_fine_data_aug_4 = get_model_4.fit(train_data,
                                                  epochs=fine_tune_epochs,
                                                  validation_data=test_data,
                                                  initial_epoch=initial_epochs,
                                                  validation_steps=int(0.25 * len(test_data)))
     # Save the fine-tuned history
    st.session_state['model_4_history_fine'] = history_fine_data_aug_4
    
    # Compare histories
    st.subheader("Training History Comparison")
    try:
        fig = compare_historys(
            original_history=initial_history,
            new_history=history_fine_data_aug_4,
            initial_epochs=initial_epochs
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not plot history comparison: {e}")

elif choice == "VLM (Vision Language Model)":
    st.subheader("Vision Language Model (CLIP) Implementation")

    # Download CLIP model
    @st.cache_resource
    def load_clip_model():
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return model, processor
        except Exception as e:
            st.error(f"Error loading CLIP model: {e}")
            return None, None

    # Load models
    clip_model, clip_processor = load_clip_model()

    if clip_model is None:
        st.error("Failed to load CLIP model. Please install transformers library.")
        st.code("pip install transformers torch")
        st.stop()

    st.success("CLIP model loaded successfully!")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Image Analysis", "Text Search", "Batch Analysis"])

    with tab1:
        st.subheader("Image-Text Matching")

        # Upload image
        uploaded_file = st.file_uploader(
            type=['jpg', 'jpeg', 'png'],
            key="vlm_image"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)

            # Text prompts
            st.subheader("Text Prompts")

            # Predefined prompts
            drug_classes = [
                'Alaxan', 'Bactidol', 'Bioflu', 'Biogesic', 'DayZinc',
                'Decolgen', 'Fish Oil', 'Kremil S', 'Medicol', 'Neozep'
            ]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Auto Prompts:**")
                auto_prompts = [f"a photo of {drug} medication" for drug in drug_classes]

                if st.button("Auto Analysis"):
                    with st.spinner("Analyzing..."):
                        try:
                            # Process with CLIP
                            inputs = clip_processor(
                                text=auto_prompts,
                                images=image,
                                return_tensors="pt",
                                padding=True
                            )

                            with torch.no_grad():
                                outputs = clip_model(**inputs)
                                logits_per_image = outputs.logits_per_image
                                probs = logits_per_image.softmax(dim=1)

                            # Show results
                            results_df = pd.DataFrame({
                                'Drug': drug_classes,
                                'Probability': [float(prob) for prob in probs[0]],
                                'Probability (%)': [f"{float(prob)*100:.1f}%" for prob in probs[0]]
                            })

                            results_df = results_df.sort_values('Probability', ascending=False)

                            # Highlight highest probability
                            st.success(
                                f"Highest probability: **{results_df.iloc[0]['Drug']}** "
                                f"({results_df.iloc[0]['Probability (%)']})"
                            )

                            # Show table
                            st.dataframe(results_df, use_container_width=True)

                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(results_df['Drug'], results_df['Probability'])
                            ax.set_title('CLIP Model Probabilities')
                            ax.set_ylabel('Probability')
                            ax.set_xlabel('Drug Classes')
                            plt.xticks(rotation=45)

                            # Color highest probability
                            bars[0].set_color('red')

                            plt.tight_layout()
                            st.pyplot(fig)

                        except Exception as e:
                            st.error(f"Analysis error: {e}")

            with col2:
                st.write("**Custom Prompt:**")
                custom_prompt = st.text_input(
                    "Text prompt:",
                    value="a photo of medicine",
                    help="Example: 'a photo of pain relief medication'"
                )

                if st.button("🔍 Custom Prompt Analysis"):
                    if custom_prompt:
                        with st.spinner("Analyzing..."):
                            try:
                                # Analyze with custom prompt
                                inputs = clip_processor(
                                    text=[custom_prompt],
                                    images=image,
                                    return_tensors="pt",
                                    padding=True
                                )

                                with torch.no_grad():
                                    outputs = clip_model(**inputs)
                                    logits_per_image = outputs.logits_per_image
                                    similarity = logits_per_image[0][0].item()

                                similarity_percent = (similarity + 1) / 2 * 100  # [-1,1] → [0,100]

                                st.metric(
                                    label="Similarity Score",
                                    value=f"{similarity_percent:.1f}%",
                                    delta=f"Raw score: {similarity:.3f}"
                                )

                                st.progress(similarity_percent / 100)

                                if similarity_percent > 70:
                                    st.success("🎯 High similarity!")
                                elif similarity_percent > 40:
                                    st.info("⚠️ Medium similarity")
                                else:
                                    st.warning("❌ Low similarity")

                            except Exception as e:
                                st.error(f"Analysis error: {e}")
                    else:
                        st.warning("Please enter a prompt!")

    with tab2:
        st.subheader("Text-Based Image Search")

        # Load dataset images
        if os.path.exists("data/Drug Vision/Data Combined"):
            search_query = st.text_input(
                "Search query:",
                value="pain relief medication",
                help="Enter the type of drug you're looking for"
            )

            num_results = st.slider("Number of results:", 1, 10, 5)

            if st.button("🔍 Search"):
                with st.spinner("Searching..."):
                    try:
                        all_images = []
                        all_paths = []

                        data_dir = "data/Drug Vision/Data Combined"
                        for class_name in os.listdir(data_dir):
                            class_path = os.path.join(data_dir, class_name)
                            if os.path.isdir(class_path):
                                for img_file in os.listdir(class_path)[:5]:
                                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                        img_path = os.path.join(class_path, img_file)
                                        try:
                                            img = Image.open(img_path).resize((224, 224))
                                            all_images.append(img)
                                            all_paths.append(img_path)
                                        except:
                                            continue

                        if all_images:
                            inputs = clip_processor(
                                text=[search_query],
                                images=all_images,
                                return_tensors="pt",
                                padding=True
                            )

                            with torch.no_grad():
                                outputs = clip_model(**inputs)
                                logits_per_image = outputs.logits_per_image
                                similarities = logits_per_image[0].cpu().numpy()

                            top_indices = np.argsort(similarities)[::-1][:num_results]

                            st.subheader(f"Top {num_results} results for '{search_query}':")

                            cols = st.columns(min(3, num_results))
                            for i, idx in enumerate(top_indices):
                                with cols[i % 3]:
                                    st.image(all_images[idx], use_column_width=True)
                                    similarity_score = (similarities[idx] + 1) / 2 * 100
                                    st.write(f"**Similarity: {similarity_score:.1f}%**")
                                    st.write(f"Path: {os.path.basename(all_paths[idx])}")

                        else:
                            st.error("No images found!")

                    except Exception as e:
                        st.error(f"Search error: {e}")
        else:
            st.warning("Dataset not found!")

    with tab3:
        st.subheader("Batch Analysis")

        uploaded_files = st.file_uploader(
            "Upload multiple images:",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="vlm_batch"
        )

        if uploaded_files:
            st.write(f"Number of uploaded images: {len(uploaded_files)}")

            batch_prompt = st.text_input(
                "Batch analysis prompt:",
                value="a photo of pharmaceutical drug",
                help="Prompt to use for all images"
            )

            if st.button("🔍 Batch Analysis"):
                with st.spinner("Analyzing batch..."):
                    try:
                        images = []
                        image_names = []

                        for uploaded_file in uploaded_files:
                            image = Image.open(uploaded_file)
                            images.append(image)
                            image_names.append(uploaded_file.name)

                        inputs = clip_processor(
                            text=[batch_prompt] * len(images),
                            images=images,
                            return_tensors="pt",
                            padding=True
                        )

                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                            similarities = outputs.logits_per_image.diagonal().cpu().numpy()

                        results_df = pd.DataFrame({
                            'Image': image_names,
                            'Similarity': similarities,
                            'Similarity (%)': [(sim + 1) / 2 * 100 for sim in similarities]
                        })

                        results_df = results_df.sort_values('Similarity', ascending=False)

                        st.subheader("📊 Batch Analysis Results")
                        st.dataframe(results_df, use_container_width=True)

                        fig, ax = plt.subplots(figsize=(12, 6))
                        bars = ax.bar(range(len(results_df)), results_df['Similarity (%)'])
                        ax.set_title('Batch Analysis Results')
                        ax.set_ylabel('Similarity (%)')
                        ax.set_xlabel('Images')
                        ax.set_xticks(range(len(results_df)))
                        ax.set_xticklabels(results_df['Image'], rotation=45, ha='right')

                        bars[0].set_color('green')

                        plt.tight_layout()
                        st.pyplot(fig)

                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="batch_analysis_results.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Batch analysis error: {e}")
        else:
            st.info("Please upload images for batch analysis!")

elif choice == "Drugs and Vitamins Detection":
    st.subheader("Drugs and Vitamins Detection")
    st.write("Upload an image to classify:")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.session_state.training_complete and st.session_state.models:
            img_array = tf.keras.utils.img_to_array(image.resize((224, 224)))
            img_array = tf.expand_dims(img_array, 0) / 255.0

            st.write("Predictions from different models:")

            if train_data is not None:
                class_names = list(train_data.class_indices.keys())

                for model_name, model in st.session_state.models.items():
                    try:
                        predictions = model.predict(img_array)
                        predicted_class = class_names[np.argmax(predictions[0])]
                        confidence = np.max(predictions[0])

                        st.write(f"**{model_name}:** {predicted_class} (Confidence: {confidence:.2%})")
                    except Exception as e:
                        st.error(f"Error with {model_name}: {e}")
        else:
            st.warning("Please train the models first.")


elif choice == "Drugs and Vitamins Detection":
    st.subheader("Drugs and Vitamins Detection")
    st.write("Upload an image to classify:")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.session_state.training_complete and st.session_state.models:
            # Preprocess image
            img_array = tf.keras.utils.img_to_array(image.resize((224, 224)))
            img_array = tf.expand_dims(img_array, 0) / 255.0
            
            # Get predictions from all models
            st.write("Predictions from different models:")
            
            if train_data is not None:
                class_names = list(train_data.class_indices.keys())
                
                for model_name, model in st.session_state.models.items():
                    try:
                        predictions = model.predict(img_array)
                        predicted_class = class_names[np.argmax(predictions[0])]
                        confidence = np.max(predictions[0])
                        
                        st.write(f"**{model_name}:** {predicted_class} (Confidence: {confidence:.2%})")
                    except Exception as e:
                        st.error(f"Error with {model_name}: {e}")
        else:
            st.warning("Please train the models first.")

elif choice == "Performance Evaluation Metrics":
    st.subheader("Performance Evaluation Metrics")

    if not st.session_state.models or not st.session_state.training_complete:
        st.warning("First, train the models.")
        st.stop()

    models_performance = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ROC-AUC Curve': []
    }

    y_true = test_data.classes

    for i, (model_name, model) in enumerate(st.session_state.models.items()):
        st.write(f"Calculating: {model_name}")
        y_pred = model.predict(test_data, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='weighted')
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')

        try:
            roc_auc_curve = roc_auc_score(
                y_true,
                y_pred,  # ehtimal vektoru!
                multi_class='ovr',
                average='weighted'
            )
        except Exception as e:
            roc_auc_curve = 0.0

        models_performance['Model'].append(model_name)
        models_performance['Accuracy'].append(accuracy)
        models_performance['Precision'].append(precision)
        models_performance['Recall'].append(recall)
        models_performance['F1 Score'].append(f1)
        models_performance['ROC-AUC Curve'].append(roc_auc_curve)

    df_perf_metrics = pd.DataFrame(models_performance)

    st.dataframe(df_perf_metrics.style.format(precision=2))

    metrics_choice = st.selectbox(
        "Select a metric:",
        ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC Curve']
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_perf_metrics['Model'], df_perf_metrics[metrics_choice], color='skyblue')
    ax.set_title(f"{metrics_choice} Comparison Across Models")
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig)
    
    

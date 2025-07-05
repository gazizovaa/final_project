import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import streamlit as st 
import tensorflow as tf
import random 
import sklearn
import io 
from helper_functions import walk_through_dir, plot_loss_curves, compare_historys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from PIL import Image 
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
    if os.path.exists('saved_models/best_model.h5'):
        return tf.keras.models.load_model('saved_models/best_model.h5')
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
            try:
                models[model_name] = tf.keras.models.load_model(path) 
                st.success(f"{model_name} loaded successfully!")
            except Exception as e:
                st.warning(f"Error loading {model_name}: {e}")
        else:
            st.info(f"{model_name} not found at {path}.")
    return models 

def save_model(model, model_name):
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/{model_name}.h5'
    model.save(model_path)
    st.success(f"Model saved to {model_path}")

def save_training_history(history, model_name):
    os.makedirs('training_history', exist_ok=True)
    history_path = f'training_history/{model_name}_history.pkl'  # Fixed typo

    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f) 

    st.success(f"Training history saved to {history_path}.")

@st.cache_data
def load_training_history(model_name):
    history_path = f'training_history/{model_name}_history.pkl'  # Fixed typo
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            return pickle.load(f)
    return None 

# Session State Management
def initialize_session_state():
    if 'models_loaded' not in st.session_state:  # Fixed typo
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

# Model creation function
def get_or_create_models():
    if not st.session_state.models_loaded:  # Fixed typo
        st.info("Loading saved models...")
        saved_models = load_multiple_models()

        if saved_models:
            st.session_state.models.update(saved_models)
            st.session_state.models_loaded = True 
            return saved_models
        else:
            st.warning("No saved models found. Models will be created when needed.")
    
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
    if not os.path.exists("data/Drug Vision/Data Combined"):
        st.error("Dataset not found. Please ensure the dataset is downloaded.")
        return None, None
    
    dir_path = "data/Drug Vision/Data Combined"
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
    
    if train_data is None or test_data is None:
        st.error("Data not loaded. Please check the dataset.")
        st.stop()
    
    models = get_or_create_models()
    
    if st.button("Train all models"):
        with st.spinner("Training Models..."):
            # Create directories
            os.makedirs('train', exist_ok=True)
            os.makedirs('test', exist_ok=True)
            os.makedirs('checkpoints', exist_ok=True)
            
            # Copy files to train/test directories
            dir_path = "data/Drug Vision/Data Combined"
            train_dir = 'train'
            test_dir = 'test'
            
            # Copy training files
            for file in train_data.filenames:
                parts = file.split('/')
                if len(parts) == 2:
                    class_name, file_name = parts 
                    src = os.path.join(dir_path, class_name, file_name)
                    train_target_dir = os.path.join(train_dir, class_name)
                    os.makedirs(train_target_dir, exist_ok=True)
                    train_dest = os.path.join(train_target_dir, file_name)
                    if os.path.exists(src) and not os.path.exists(train_dest):
                        shutil.copy2(src, train_dest)

            # Copy test files
            for file in test_data.filenames:
                parts = file.split('/')
                if len(parts) == 2:
                    class_name, file_name = parts
                    src = os.path.join(dir_path, class_name, file_name)
                    test_target_dir = os.path.join(test_dir, class_name)
                    os.makedirs(test_target_dir, exist_ok=True)
                    test_dest = os.path.join(test_target_dir, file_name)
                    if os.path.exists(src) and not os.path.exists(test_dest):
                        shutil.copy2(src, test_dest)
            
            # Train models
            for model_name, model in models.items():
                st.write(f"Training {model_name}...")
                
                # Compile model
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
                    callbacks=[model_ckp, model_es]
                )
                
                # Save model and history
                save_model(model, model_name)
                save_training_history(history, model_name)
                
                # Store in session state
                st.session_state[f"{model_name}_history"] = history.history
            
            st.session_state.training_complete = True
            st.success("All models trained successfully!")
    
    # Display model summary for model_1
    if 'model_1' in models:
        model_1 = models['model_1']
        
        # Model summary
        stream = io.StringIO()
        model_1.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        st.text(summary_string)
        
        # Model layers table
        layers = [(layer.name, str(layer.output.shape), layer.count_params()) for layer in model_1.layers]
        df_layers = pd.DataFrame(layers, columns=["Layer name", "Output shape", "Param #"])
        st.table(df_layers)
        
        # Evaluate model if trained
        if st.session_state.training_complete:
            loss, accuracy = model_1.evaluate(test_data, verbose=0)
            st.write(f"Test Loss: {loss:.4f}")
            st.write(f"Test Accuracy: {accuracy:.4f}")
            
            # Plot training history
            if 'model_1_history' in st.session_state:
                fig = plot_loss_curves(st.session_state['model_1_history'])
                st.pyplot(fig)

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
        classes = list(class_counts.keys())  # Fixed typo
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
    st.write("This section would show fine-tuning results.")
    st.info("Fine-tuning functionality can be implemented after basic training is complete.")

elif choice == "VLM (Vision Language Model)":
    st.subheader("Vision Language Model (CLIP) Implementation")
    st.warning("CLIP functionality requires additional setup and dependencies.")
    st.info("This section would implement CLIP-based image-text matching.")

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

    # Create a dictionary for metrics metadata
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
        st.write(f"Hesablanır: {model_name}")
        y_pred = model.predict(test_data)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Metrics
        accuracy = accuracy_score(y_true, y_pred_classes) 
        precision = precision_score(y_true, y_pred_classes)
        recall = recall_score(y_true, y_pred_classes)
        f1 = f1_score(y_true, y_pred_classes)
        roc_auc_curve = roc_auc_score(y_true, y_pred_classes)

        models_performance['Model'].append(model_name)  
        models_performance['Accuracy'].append(accuracy)
        models_performance['Precision'].append(precision)
        models_performance['Recall'].append(recall)
        models_performance['F1 Score'].append(f1)
        models_performance['ROC-AUC Curve'].append(roc_auc_curve) 
    
    # Create dataframe to save metrics information
    df_perf_metrics = pd.DataFrame(models_performance)

    # Show the dataframe as a table
    st.dataframe(df_perf_metrics.style.format(precision=2))

    # Create a selectbox
    metrics_choice = st.selectbox("Select a metric:",
                                  ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC Curve'])
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_perf_metrics['Model'], df_perf_metrics[metrics_choice], color='skyblue')
    ax.set_title(f"{metrics_choice} Comparison Across Models")
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy = (bar.get_x() + bar.get_width() / 2, height),
                    xytext = (0, 3),
                    textcoords = "offset points",
                    ha = "center", var = "bottom")
        
    plt.tight_layout()
    st.pyplot(fig) 
        
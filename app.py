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

st.set_page_config(
    page_title="Pharmaceutical Drugs and Vitamins Detection",
    layout="wide",
    initial_sidebar_state="auto"
)

@st.cache_resource
def load_multiple_models():
    models = {}
    model_paths = {
        'CNN Model': 'saved_models/cnn_model.h5',
        'InceptionV3': 'saved_models/inception_model.h5',
        'MobileNetV2': 'saved_models/mobilenet_model.h5',
        'EfficientNet': 'saved_models/efficientnet_model.h5'
    }

    for model_name, path in model_paths.items():
        try:
            if os.path.exists(path):
                models[model_name] = tf.keras.models.load_model(path)
                st.success(f"✅ {model_name} successfully loaded!")
            else:
                st.info(f"ℹ️ {model_name} model not found at '{path}'. Please ensure model files are available.")
        except Exception as e:
            st.error(f"❌ Error loading {model_name}: {e}")
    return models

@st.cache_resource
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
    st.success(f"💾 Model saved to '{model_path}'.")

def save_training_history(history_dict, model_name):
    os.makedirs('training_history', exist_ok=True)
    history_path = f'training_history/{model_name}_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history_dict, f)
    st.success(f"📈 Training history saved to '{history_path}'.")

def initialize_session_state():
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
        st.session_state.models = {}
        st.session_state.training_complete = False
        st.session_state.data_loaded = False

initialize_session_state()

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomHeight(0.2),
    tf.keras.layers.RandomWidth(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

def download_dataset():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        if not os.path.exists("data"):
            os.makedirs("data")

        if not os.path.exists("data/Drug Vision/Data Combined"):
            if "kaggle_username" in st.secrets and "kaggle_key" in st.secrets:
                os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle_username"]
                os.environ["KAGGLE_KEY"] = st.secrets["kaggle_key"]
                api = KaggleApi()
                api.authenticate()
                api.dataset_download_files("vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images",
                                            path="data", unzip=True)
                st.success("Dataset successfully downloaded and unzipped.")
            else:
                st.warning("Kaggle API keys not found. Please configure your `st.secrets` file or download the dataset manually.")
        else:
            st.info("Dataset already exists.")
    except Exception as e:
        st.warning(f"Dataset download failed: {e}. Please ensure your Kaggle API keys are correct or that the dataset exists.")

def create_train_and_test_directories():
    train_dir = 'train'
    test_dir = 'test'

    st.info("Deleting and recreating train/test directories...")
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    dir_path = "data/Drug Vision/Data Combined"
    if not os.path.exists(dir_path):
        st.error(f"Source directory '{dir_path}' not found for copying files.")
        return

    st.warning("`create_train_and_test_directories` function is not optimized for deployment and its purpose needs clarification.")

download_dataset()

def get_or_create_models():
    if not st.session_state.models:
        input_shape = (224, 224, 3)

        model_1 = tf.keras.Sequential([
            data_augmentation,
            tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', input_shape=input_shape, activation='relu'),
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
        ], name="CNN_Model")
        model_1.build(input_shape=(None, *input_shape))

        base_model_inception = tf.keras.applications.InceptionV3(include_top=False,
                                                       input_shape=input_shape,
                                                       weights='imagenet')
        base_model_inception.trainable = False
        inputs_inception = tf.keras.layers.Input(shape=input_shape, name='input_layer_inception')
        x_inception = data_augmentation(inputs_inception)
        x_inception = base_model_inception(x_inception, training=False)
        x_inception = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling_layer_inception')(x_inception)
        outputs_inception = tf.keras.layers.Dense(10, activation='softmax', name='output_layer_inception')(x_inception)
        model_2 = tf.keras.Model(inputs_inception, outputs_inception, name="InceptionV3")
        model_2.build(input_shape=(None, *input_shape))

        base_model_mobilenet = tf.keras.applications.MobileNetV2(include_top=False,
                                                       input_shape=input_shape,
                                                       weights='imagenet')
        base_model_mobilenet.trainable = False
        inputs_mobilenet = tf.keras.layers.Input(shape=input_shape, name='input_layer_mobilenet')
        x_mobilenet = data_augmentation(inputs_mobilenet)
        x_mobilenet = base_model_mobilenet(x_mobilenet, training=False)
        x_mobilenet = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling_layer_mobilenet')(x_mobilenet)
        outputs_mobilenet = tf.keras.layers.Dense(10, activation='softmax', name='output_layer_mobilenet')(x_mobilenet)
        model_3 = tf.keras.Model(inputs_mobilenet, outputs_mobilenet, name="MobileNetV2")
        model_3.build(input_shape=(None, *input_shape))

        base_model_efficientnet = tf.keras.applications.EfficientNetV2B0(include_top=False,
                                                           input_shape=input_shape,
                                                           weights='imagenet')
        base_model_efficientnet.trainable = False
        inputs_efficientnet = tf.keras.layers.Input(shape=input_shape, name='input_layer_efficientnet')
        x_efficientnet = data_augmentation(inputs_efficientnet)
        x_efficientnet = base_model_efficientnet(x_efficientnet, training=False)
        x_efficientnet = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling_layer_efficientnet')(x_efficientnet)
        outputs_efficientnet = tf.keras.layers.Dense(10, activation='softmax', name='output_layer_efficientnet')(x_efficientnet)
        model_4 = tf.keras.Model(inputs_efficientnet, outputs_efficientnet, name="EfficientNetV2")
        model_4.build(input_shape=(None, *input_shape))

        st.session_state.models = {
            'CNN Model': model_1,
            'InceptionV3': model_2,
            'MobileNetV2': model_3,
            'EfficientNet': model_4
        }
    return st.session_state.models


@st.cache_resource
def load_data_for_training():
    dir_path = "data/Drug Vision/Data Combined"
    if not os.path.exists(dir_path):
        st.error(f"Dataset path not found: {dir_path}. Please ensure the dataset exists.")
        return None, None, []

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    tf.random.set_seed(42)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(directory=dir_path,
                                                   target_size=IMG_SIZE,
                                                   batch_size=BATCH_SIZE,
                                                   class_mode='categorical',
                                                   subset='training',
                                                   seed=42)

    test_data = train_datagen.flow_from_directory(directory=dir_path,
                                                  target_size=IMG_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical',
                                                  subset='validation',
                                                  seed=42)

    class_names = list(train_data.class_indices.keys())
    return train_data, test_data, class_names

train_data, test_data, class_names = load_data_for_training()
if not class_names:
    class_names = ['Alaxan', 'Bactidol', 'Bioflu', 'Biogesic', 'DayZinc', 'Decolgen', 'Fish Oil', 'Kremil S', 'Medicol', 'Neozep']

# Define data_dir_clip_search at a higher scope
data_dir_clip_search = "data/Drug Vision/Data Combined"


st.title("💊 Pharmaceutical Drugs and Vitamins Classification App")

choice = st.sidebar.radio("Navigation", ["Dataset Overview", "Model Training", "Data Exploration",
                                         "Data Augmentation", "Transfer Learning", "Fine Tuning",
                                         "Vision Language Model", "Drugs and Vitamins Detection",
                                         "Performance Metrics"])

if choice == "Dataset Overview":
    st.subheader("📊 Dataset Overview")

    data_overview_path = "data/Drug Vision"
    if os.path.exists(data_overview_path):
        st.text(walk_through_dir(data_overview_path))

        combined_data_path = "data/Drug Vision/Data Combined"
        if os.path.exists(combined_data_path):
            num_of_classes = len(os.listdir(combined_data_path))
            st.markdown(f"Number of classes: **{num_of_classes}**")

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
                        st.write(f'Image shape: {img.shape}')
                        st.pyplot(fig)
            if class_names:
                 view_random_image(target_dir=combined_data_path, target_classes=class_names)
            else:
                 st.warning("Class names not found. Dataset might not exist or class names are not defined.")
        else:
            st.error(f"Combined dataset path '{combined_data_path}' not found. Please ensure the dataset exists.")
    else:
        st.error(f"Dataset path '{data_overview_path}' not found. Please ensure the dataset is downloaded.")

elif choice == "Model Training":
    st.subheader("🏋️‍♀️ Model Training")
    st.info("""
        The model training process will not provide direct "terminal" output in Streamlit.
        However, you can monitor the progress with a progress bar as each model trains,
        and after training is complete, you will see accuracy and loss results for each epoch in a table format.
        If models have been previously trained, you can retrain them in this section.
    """)

    models = get_or_create_models()
    model_tabs = st.tabs([f"Model: {model_name}" for model_name in models.keys()])

    if st.button("Train all models", key="train_all_models_button"):
        if train_data is None or test_data is None:
            st.error("Dataset is not ready for training. Please ensure the dataset is loaded.")
        else:
            with st.spinner("Models are being trained... This may take a while."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (model_name, model) in enumerate(models.items()):
                    status_text.text(f"⏳ Training {model_name} ({i+1}/{len(models)})...")

                    model.compile(loss='categorical_crossentropy',
                                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                  metrics=['accuracy'])

                    checkpoint_path = f'checkpoints/{model_name}_checkpoint.weights.h5'
                    os.makedirs('checkpoints', exist_ok=True)
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

                    history = model.fit(
                        train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data),
                        callbacks=[model_ckp, model_es],
                        verbose=0
                    )

                    save_model(model, model_name)
                    save_training_history(history.history, model_name)

                    st.session_state[f"{model_name}_history"] = history.history
                    progress_bar.progress((i + 1) / len(models))

                st.session_state.training_complete = True
                st.success("✅ All models successfully trained!")
                st.balloons()

    for (model_name, model), tab in zip(models.items(), model_tabs):
        with tab:
            st.write(f"#### {model_name} Summary")

            with st.expander("Model Architecture", expanded=False):
                stream = io.StringIO()
                model.summary(print_fn=lambda x: stream.write(x + '\n'))
                st.text(stream.getvalue())

                layers = [(layer.name, str(layer.output.shape), layer.count_params())
                          for layer in model.layers]
                st.table(pd.DataFrame(layers,
                                      columns=["Layer Name", "Output Shape", "Number of Parameters"]))

            history_dict = st.session_state.get(f"{model_name}_history")
            if history_dict is None:
                history_dict = load_training_history(model_name)

            if history_dict:
                st.write("##### Training Performance")

                epochs_df = pd.DataFrame(history_dict)
                epochs_df.index.name = 'Epoch'
                epochs_df.index = epochs_df.index + 1

                st.dataframe(epochs_df.style.format({
                    'accuracy': '{:.2%}',
                    'loss': '{:.4f}',
                    'val_accuracy': '{:.2%}',
                    'val_loss': '{:.4f}'
                }))

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Last Training Accuracy", f"{history_dict['accuracy'][-1]:.2%}")
                    st.metric("Last Training Loss", f"{history_dict['loss'][-1]:.4f}")
                with col2:
                    st.metric("Last Validation Accuracy", f"{history_dict['val_accuracy'][-1]:.2%}")
                    st.metric("Last Validation Loss", f"{history_dict['val_loss'][-1]:.4f}")

                dummy_history_obj = type('obj', (object,), {'history': history_dict})()
                fig_loss_curves = plot_loss_curves(dummy_history_obj)
                st.pyplot(fig_loss_curves)

                if st.button(f"Evaluate {model_name} model", key=f"evaluate_{model_name}"):
                    if test_data is None:
                        st.error("Test data is not available. Please ensure the dataset is loaded.")
                    else:
                        loss, accuracy = model.evaluate(test_data, verbose=0)
                        st.success(f"📈 {model_name} Test Accuracy: **{accuracy:.2%}**")
                        st.info(f"Test Loss: **{loss:.4f}**")
            else:
                st.warning("Model has not been trained yet or training history not found. Please click 'Train all models' button.")

elif choice == "Data Exploration":
    st.subheader("🔍 Data Exploration")
    data_dir = "data/Drug Vision/Data Combined"

    if os.path.exists(data_dir):
        class_counts = {}
        total_images = 0

        for class_name_in_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name_in_dir)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_name_in_dir] = count
                total_images += count

        fig, ax = plt.subplots(figsize=(12, 6))
        classes_data_exploration = list(class_counts.keys())
        counts = list(class_counts.values())

        ax.bar(classes_data_exploration, counts)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Images')
        ax.set_title('Image Distribution by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.write(f"**Total Images:** {total_images}")
        st.write(f"**Number of Classes:** {len(classes_data_exploration)}")
    else:
        st.error(f"Dataset path '{data_dir}' not found. Please ensure the dataset is downloaded.")

elif choice == "Data Augmentation":
    st.subheader("Data Augmentation")

    sample_image_path_root = "data/Drug Vision/Data Combined"
    if os.path.exists(sample_image_path_root):
        class_folders_aug = os.listdir(sample_image_path_root)
        selected_class_aug = st.selectbox("Select a class:", class_folders_aug)
        class_path_aug = os.path.join(sample_image_path_root, selected_class_aug)

        if os.path.exists(class_path_aug):
            images_aug = [f for f in os.listdir(class_path_aug) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images_aug:
                sample_img_path = os.path.join(class_path_aug, images_aug[0])
                original_img = tf.keras.utils.load_img(sample_img_path, target_size=(224, 224))
                img_arr = tf.keras.utils.img_to_array(original_img)
                img_arr = tf.expand_dims(img_arr, 0)

                st.write("Original image and augmented versions:")
                fig, axes = plt.subplots(2, 4, figsize=(15, 8))
                axes = axes.flatten()

                axes[0].imshow(original_img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                for i in range(1, 8):
                    augmented_img = data_augmentation(img_arr)
                    axes[i].imshow(tf.keras.utils.array_to_img(augmented_img[0]))
                    axes[i].set_title(f"Augmented {i}")
                    axes[i].axis("off")

                st.pyplot(fig)
            else:
                st.error(f"No images found in the selected class '{selected_class_aug}'.")
        else:
            st.error(f"Class path '{class_path_aug}' not found.")
    else:
        st.error(f"Dataset path '{sample_image_path_root}' not found. Please ensure the dataset is downloaded.")

elif choice == "Transfer Learning":
    st.subheader("🌐 Transfer Learning Results")
    st.write("""
        This section is designed to compare the training results of pre-trained models such as InceptionV3, MobileNetV2, and EfficientNet
        for drug classification. The table shows the final accuracy and loss achieved by each model during training.
    """)

    transfer_learning_models = ['InceptionV3', 'MobileNetV2', 'EfficientNet']

    if st.session_state.training_complete:
        results = []
        for model_name in transfer_learning_models:
            history_dict = st.session_state.get(f"{model_name}_history")
            if history_dict is None:
                history_dict = load_training_history(model_name)

            if history_dict:
                last_accuracy = history_dict['accuracy'][-1]
                last_loss = history_dict['loss'][-1]
                last_val_accuracy = history_dict['val_accuracy'][-1]
                last_val_loss = history_dict['val_loss'][-1]
                results.append({
                    'Model': model_name,
                    'Last Training Accuracy': last_accuracy,
                    'Last Training Loss': last_loss,
                    'Last Validation Accuracy': last_val_accuracy,
                    'Last Validation Loss': last_val_loss
                })
            else:
                st.warning(f"Training history not found for model '{model_name}'.")

        if results:
            df_results = pd.DataFrame(results)
            st.write("#### Summary Results of Transfer Learning Models")
            st.dataframe(df_results.style.format({
                'Last Training Accuracy': '{:.2%}',
                'Last Training Loss': '{:.4f}',
                'Last Validation Accuracy': '{:.2%}',
                'Last Validation Loss': '{:.4f}'
            }), use_container_width=True)

            fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
            bars_perf = ax_perf.bar(df_results['Model'], df_results['Last Validation Accuracy'], color=['skyblue', 'lightcoral', 'lightgreen'])
            ax_perf.set_title('Validation Accuracy of Transfer Learning Models')
            ax_perf.set_ylabel('Accuracy')
            ax_perf.set_ylim(0, 1)
            plt.xticks(rotation=45)

            for bar in bars_perf:
                yval = bar.get_height()
                ax_perf.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2%}', ha='center', va='bottom')

            st.pyplot(fig_perf)

        else:
            st.info("No results available for Transfer Learning models. Please train the models in the 'Model Training' section first.")
    else:
        st.info("Please train the models in the 'Model Training' section first to see the results here.")

elif choice == "Fine Tuning":
    st.subheader("Fine Tuning")

    if not st.session_state.models or not st.session_state.training_complete:
        st.warning("Please train the models in the 'Model Training' section first.")
        st.stop()

    model_options = list(st.session_state.models.keys())[1:]
    selected_model_ft = st.selectbox("Select a model for fine-tuning:", model_options)

    model_ft = st.session_state.models[selected_model_ft]
    history_ft_initial = st.session_state.get(f"{selected_model_ft}_history")

    if history_ft_initial is None:
        st.warning(f"Training history not found for the selected model '{selected_model_ft}'. Please train it first.")
        st.stop()

    st.write(f"### Fine-Tuning of {selected_model_ft} Model")

    with st.expander("Model Layers", expanded=False):
        for layer_number, layer in enumerate(model_ft.layers):
            st.write(f"Layer {layer_number}: {layer.name} | Trainable: {layer.trainable}")

    st.write("### Fine-Tuning Options")
    unfreeze_layers = st.number_input("Number of layers to unfreeze:", min_value=1, max_value=100, value=10, key=f"unfreeze_{selected_model_ft}")
    learning_rate_ft = st.number_input("Learning rate for fine-tuning:", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.6f", key=f"lr_ft_{selected_model_ft}")
    epochs_ft = st.number_input("Additional number of epochs:", min_value=1, max_value=20, value=5, key=f"epochs_ft_{selected_model_ft}")

    if st.button("Start Fine-Tuning", key=f"start_ft_{selected_model_ft}"):
        if train_data is None or test_data is None:
            st.error("Dataset is not ready for fine-tuning. Please ensure the dataset is loaded.")
        else:
            with st.spinner(f"⏳ Fine-tuning {selected_model_ft}..."):
                try:
                    base_model = model_ft.layers[2]

                    base_model.trainable = True
                    for layer in base_model.layers[:-unfreeze_layers]:
                        layer.trainable = False

                    model_ft.compile(loss='categorical_crossentropy',
                                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_ft),
                                  metrics=['accuracy'])

                    initial_epochs = len(history_ft_initial['loss'])
                    fine_tune_epochs = initial_epochs + epochs_ft

                    fine_history = model_ft.fit(
                        train_data,
                        epochs=fine_tune_epochs,
                        initial_epoch=initial_epochs,
                        validation_data=test_data,
                        validation_steps=len(test_data),
                        verbose=0
                    )

                    st.session_state[f"{selected_model_ft}_history_fine"] = fine_history.history
                    save_training_history(fine_history.history, f"{selected_model_ft}_fine_tuned")

                    st.success("✅ Fine-tuning completed!")
                    st.write("### Training History Comparison")

                    dummy_history_original = type('obj', (object,), {'history': history_ft_initial})()
                    dummy_history_new = type('obj', (object,), {'history': fine_history.history})()

                    fig_compare = compare_historys(
                        original_history=dummy_history_original,
                        new_history=dummy_history_new,
                        initial_epochs=initial_epochs
                    )
                    st.pyplot(fig_compare)

                except Exception as e:
                    st.error(f"❌ An error occurred during fine-tuning: {e}")
                    st.info("Please ensure the dataset is loaded correctly and models are defined properly.")

elif choice == "Vision Language Model":
    st.subheader("Vision Language Model (CLIP) Application")

    @st.cache_resource
    def load_clip_model():
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return model, processor
        except Exception as e:
            st.error(f"❌ Error loading CLIP model: {e}. Please ensure 'transformers' and 'torch' libraries are installed.")
            return None, None

    clip_model, clip_processor = load_clip_model()

    if clip_model is None:
        st.error("CLIP model not loaded. Please install the 'transformers' library.")
        st.code("pip install transformers torch")
        st.stop()

    st.success("✅ CLIP model successfully loaded!")

    tab1, tab2 = st.tabs(["Image Classification", "Text-Based Search"])

    with tab1:
        st.subheader("Image Classification with CLIP")

        uploaded_file_clip = st.file_uploader(
            "Upload an image of a pharmaceutical product:",
            type=['jpg', 'jpeg', 'png'],
            key="vlm_image"
        )

        if uploaded_file_clip is not None:
            image_clip = Image.open(uploaded_file_clip)
            st.image(image_clip, caption="Uploaded image", use_column_width=True)

            drug_classes = class_names

            prompts = [f"a photo of {drug} medication" for drug in drug_classes]

            if st.button("Analyze with CLIP", key="analyze_clip"):
                with st.spinner("Analyzing..."):
                    try:
                        inputs = clip_processor(
                            text=prompts,
                            images=image_clip,
                            return_tensors="pt",
                            padding=True
                        )

                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            probs = logits_per_image.softmax(dim=1)

                        results_df = pd.DataFrame({
                            'Drug': drug_classes,
                            'Probability': [float(prob) for prob in probs[0]],
                            'Probability (%)': [f"{float(prob)*100:.1f}%" for prob in probs[0]]
                        })

                        results_df = results_df.sort_values('Probability', ascending=False)

                        st.success(
                            f"✨ Highest probability: **{results_df.iloc[0]['Drug']}** "
                            f"({results_df.iloc[0]['Probability (%)']})"
                        )

                        st.dataframe(results_df, use_container_width=True)

                        fig_clip_probs, ax_clip_probs = plt.subplots(figsize=(10, 6))
                        bars_clip = ax_clip_probs.bar(results_df['Drug'], results_df['Probability'])
                        ax_clip_probs.set_title('CLIP Model Probabilities')
                        ax_clip_probs.set_ylabel('Probability')
                        ax_clip_probs.set_xlabel('Drug Classes')
                        plt.xticks(rotation=45)

                        bars_clip[0].set_color('red')

                        plt.tight_layout()
                        st.pyplot(fig_clip_probs)

                    except Exception as e:
                        st.error(f"❌ Analysis error: {e}")

    with tab2:
        st.subheader("Text-Based Image Search")

        # data_dir_clip_search is now defined at a higher scope
        if os.path.exists(data_dir_clip_search):
            search_query_clip = st.text_input(
                "Search for drug type:",
                value="painkiller drug",
                help="Enter the type of drug you are looking for",
                key="clip_search_query"
            )

            num_results_clip = st.slider("Number of results to display:", 1, 10, 5, key="clip_num_results")

            if st.button("Search in dataset", key="search_dataset_clip"):
                with st.spinner("Searching..."):
                    try:
                        all_images = []
                        all_paths = []
                        all_classes = []

                        for class_name_in_dir in os.listdir(data_dir_clip_search):
                            class_path_clip_search = os.path.join(data_dir_clip_search, class_name_in_dir)
                            if os.path.isdir(class_path_clip_search):
                                for img_file_clip_search in os.listdir(class_path_clip_search)[:20]:
                                    if img_file_clip_search.lower().endswith(('.jpg', '.jpeg', '.png')):
                                        img_path_clip_search = os.path.join(class_path_clip_search, img_file_clip_search)
                                        try:
                                            img_clip_search = Image.open(img_path_clip_search).resize((224, 224))
                                            all_images.append(img_clip_search)
                                            all_paths.append(img_path_clip_search)
                                            all_classes.append(class_name_in_dir)
                                        except Exception as img_err:
                                            st.warning(f"Error loading image: {img_path_clip_search} - {img_err}")
                                            continue

                        if all_images:
                            batch_size_clip = 32
                            similarities = []

                            for i in range(0, len(all_images), batch_size_clip):
                                batch_images = all_images[i:i+batch_size_clip]
                                inputs = clip_processor(
                                    text=[search_query_clip],
                                    images=batch_images,
                                    return_tensors="pt",
                                    padding=True
                                )

                                with torch.no_grad():
                                    outputs = clip_model(**inputs)
                                    batch_similarities = outputs.logits_per_image[0].cpu().numpy()
                                    similarities.extend(batch_similarities)

                            top_indices = np.argsort(similarities)[::-1][:num_results_clip]

                            st.subheader(f"Top {num_results_clip} results for '{search_query_clip}':")

                            cols_clip = st.columns(min(3, num_results_clip))
                            for i, idx in enumerate(top_indices):
                                with cols_clip[i % 3]:
                                    st.image(all_images[idx], use_column_width=True, caption=all_classes[idx])
                                    similarity_score_percentage = (similarities[idx] - np.min(similarities)) / (np.max(similarities) - np.min(similarities)) * 100
                                    st.write(f"**Similarity:** {similarity_score_percentage:.2f}%")
                                    st.write(f"**Class:** {all_classes[idx]}")
                        else:
                            st.warning("No images found for search.")
                    except Exception as e:
                        st.error(f"❌ An error occurred during search: {e}")
        else:
            st.error(f"Dataset path '{data_dir_clip_search}' not found. Please ensure the dataset is downloaded.")

elif choice == "Drugs and Vitamins Detection":
    st.subheader("Drugs and Vitamins Detection")
    st.write("Upload an image to classify it using our trained models:")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.session_state.training_complete and st.session_state.models:
            img_array = tf.keras.utils.img_to_array(image.resize((224, 224)))
            img_array = tf.expand_dims(img_array, 0) / 255.0

            st.write("### Predictions from different models:")

            if class_names:
                results = []

                for model_name, model in st.session_state.models.items():
                    try:
                        predictions = model.predict(img_array)
                        predicted_class = class_names[np.argmax(predictions[0])]
                        confidence = np.max(predictions[0])

                        results.append({
                            "Model": model_name,
                            "Prediction": predicted_class,
                            "Confidence": f"{confidence:.2%}"
                        })
                    except Exception as e:
                        st.error(f"Error during prediction with {model_name}: {e}")
                        results.append({
                            "Model": model_name,
                            "Prediction": "Error",
                            "Confidence": "N/A"
                        })
                st.dataframe(pd.DataFrame(results))
            else:
                st.error("Class names are not defined. Please ensure class names are available for prediction.")
        else:
            st.warning("Models are not trained or loaded. Please train them in the 'Model Training' section first.")

elif choice == "Performance Metrics":
    st.subheader("Performance Metrics")
    st.write("Compare the performance metrics of the trained models.")

    if not st.session_state.training_complete:
        st.warning("Please train the models in the 'Model Training' section first.")
        st.stop()

    models_performance = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ROC-AUC': []
    }

    if train_data is None or test_data is None:
        st.error("Dataset is not ready for testing. Please ensure the dataset is loaded.")
        st.stop()

    st.info("Calculating performance metrics requires loading models and test data. This can be memory-intensive in some cases.")

    for model_name, model in st.session_state.models.items():
        st.write(f"**{model_name}** metrics are being calculated...")
        y_pred_probs = model.predict(test_data, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        y_true = []
        for i in range(len(test_data)):
            _, labels = test_data[i]
            y_true.extend(np.argmax(labels, axis=1))
        y_true = np.array(y_true)

        if len(y_true) != len(y_pred):
            st.warning(f"Prediction and true label lengths for {model_name} do not match. May not have processed the entire test dataset.")
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            y_pred_probs = y_pred_probs[:min_len]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        try:
            roc_auc = roc_auc_score(
                y_true,
                y_pred_probs,
                multi_class='ovr',
                average='weighted'
            )
        except Exception as e:
            st.warning(f"An error occurred while calculating ROC-AUC ({model_name}): {e}. Likely due to class distribution or incompatible predictions.")
            roc_auc = 0.0

        models_performance['Model'].append(model_name)
        models_performance['Accuracy'].append(accuracy)
        models_performance['Precision'].append(precision)
        models_performance['Recall'].append(recall)
        models_performance['F1 Score'].append(f1)
        models_performance['ROC-AUC'].append(roc_auc)

    df_perf_metrics = pd.DataFrame(models_performance)

    st.dataframe(df_perf_metrics.style.format({
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1 Score': '{:.2%}',
        'ROC-AUC': '{:.2%}'
    }))

    metrics_choice = st.selectbox(
        "Select a metric to visualize:",
        ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_perf_metrics['Model'], df_perf_metrics[metrics_choice], color='skyblue')
    ax.set_title(f"{metrics_choice} Comparison Across Models")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

    st.pyplot(fig)
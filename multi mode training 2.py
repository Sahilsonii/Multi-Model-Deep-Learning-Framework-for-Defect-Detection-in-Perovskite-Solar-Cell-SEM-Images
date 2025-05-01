import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, LayerNormalization, Conv2D, Reshape, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import InceptionV3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
from datetime import datetime
import logging
import json
from tqdm import tqdm
import time
from matplotlib.backends.backend_pdf import PdfPages
import sys
from pathlib import Path
from config import process_image_for_prediction
import gc

# Enable mixed precision training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Custom Mish activation
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

# Register mish as a function
tf.keras.utils.get_custom_objects().update({'mish': mish})

# Import configurations
from config import (
    IMAGE_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, DEFECT_CLASSES, MODEL_DIR, EPOCHS, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, TRANSFER_LEARNING, FINE_TUNING_EPOCHS,
    FINE_TUNING_LEARNING_RATE, LOSS, METRICS, TRAIN_DIR, VALIDATION_DIR,
    TEST_DIR, LEARNING_RATE, VISUALIZATION_DIR
)

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PerovskiteMultiModelTrainer")

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Model-specific batch sizes
model_batch_sizes = {
    'VisionTransformer': BATCH_SIZE,
    'CoCa': BATCH_SIZE,
    'YOLOv9': BATCH_SIZE,
    'InceptionV3': 1
}

# Custom layer for class token
class ClassTokenLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super(ClassTokenLayer, self).__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, self.hidden_size),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_tokens = tf.repeat(self.class_token, repeats=batch_size, axis=0)
        return tf.concat([class_tokens, inputs], axis=1)

    def get_config(self):
        config = super(ClassTokenLayer, self).get_config()
        config.update({'hidden_size': self.hidden_size})
        return config

# CSP block for YOLOv9
def csp_block(x, filters, num_blocks=1):
    shortcut = Conv2D(filters, 1, padding='same', activation=None)(x)
    shortcut = BatchNormalization()(shortcut)
    shortcut = tf.keras.layers.Activation(mish)(shortcut)
    main = Conv2D(filters // 2, 1, padding='same', activation=None)(x)
    main = BatchNormalization()(main)
    main = tf.keras.layers.Activation(mish)(main)
    for _ in range(num_blocks):
        residual = main
        main = Conv2D(filters // 2, 1, padding='same', activation=None)(main)
        main = BatchNormalization()(main)
        main = tf.keras.layers.Activation(mish)(main)
        main = Conv2D(filters // 2, 3, padding='same', activation=None)(main)
        main = BatchNormalization()(main)
        main = tf.keras.layers.Activation(mish)(main)
        main = tf.keras.layers.add([residual, main])
    main = Conv2D(filters // 2, 1, padding='same', activation=None)(main)
    main = BatchNormalization()(main)
    main = tf.keras.layers.Activation(mish)(main)
    x = tf.keras.layers.concatenate([main, shortcut])
    x = Conv2D(filters, 1, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation(mish)(x)
    return x

# Residual block for CoCa
def residual_block(x, filters, downsample=False):
    residual = x
    stride = 2 if downsample else 1
    x = Conv2D(filters, 3, strides=stride, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    if downsample or residual.shape[-1] != filters:
        residual = Conv2D(filters, 1, strides=stride, padding='same')(residual)
        residual = BatchNormalization()(residual)
    x = tf.keras.layers.add([x, residual])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Learning rate schedule
def get_lr_schedule(train_generator):
    initial_learning_rate = LEARNING_RATE
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        decay_steps=EPOCHS * (train_generator.samples // model_batch_sizes['VisionTransformer']),
        alpha=0.01
    )
    return lr_schedule

# YOLOv9 model
def create_yolov9_backbone(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, strides=2, padding='same', activation=None)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation(mish)(x)
    x = Conv2D(64, 3, strides=2, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation(mish)(x)
    x = csp_block(x, 64, num_blocks=2)
    x = Conv2D(128, 3, strides=2, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation(mish)(x)
    x = csp_block(x, 128, num_blocks=6)
    x = Conv2D(256, 3, strides=2, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation(mish)(x)
    x = csp_block(x, 512, num_blocks=6)
    x = csp_block(x, 1024, num_blocks=3)
    spatial_attention = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    spatial_attention = Conv2D(1024, 1, padding='same', activation='sigmoid')(spatial_attention)
    x = tf.keras.layers.multiply([x, spatial_attention])
    return tf.keras.Model(inputs, x, name='yolov9_backbone')

def create_yolov9_model(input_shape=(224, 224, 3), num_classes=len(DEFECT_CLASSES)):
    backbone = create_yolov9_backbone(input_shape)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs=backbone.input, outputs=outputs, name='yolov9_classifier')

# Vision Transformer
def create_vit_backbone(input_shape, patch_size=16, num_heads=1, transformer_layers=1, hidden_size=64):
    inputs = Input(shape=input_shape)
    patch_dim = patch_size * patch_size * input_shape[2]
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    patches = Conv2D(hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    patches = Reshape((num_patches, hidden_size))(patches)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=hidden_size)(positions)
    x = patches + pos_embed
    x = ClassTokenLayer(hidden_size=hidden_size)(x)
    for _ in range(transformer_layers):
        norm1 = LayerNormalization(epsilon=1e-6)(x)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size // num_heads
        )(norm1, norm1)
        x = tf.keras.layers.add([attention_output, x])
        norm2 = LayerNormalization(epsilon=1e-6)(x)
        mlp = Dense(hidden_size * 4, activation='gelu')(norm2)
        mlp = Dense(hidden_size)(mlp)
        x = tf.keras.layers.add([mlp, x])
    x = LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0]
    return tf.keras.Model(inputs, x, name='vit_backbone')

def create_vit_model(input_shape=(224, 224, 3), num_classes=len(DEFECT_CLASSES)):
    base = create_vit_backbone(input_shape)
    x = base.output
    x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs=base.input, outputs=outputs, name='vit_classifier')

# CoCa model
def create_coca_backbone(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, strides=2, padding='same', activation=None)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)
    x = residual_block(x, 512)
    context = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    context = Conv2D(512, 1, padding='same', activation='sigmoid')(context)
    x = tf.keras.layers.multiply([x, context])
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(768)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return tf.keras.Model(inputs, x, name='coca_backbone')

def create_coca_model(input_shape=(224, 224, 3), num_classes=len(DEFECT_CLASSES)):
    base = create_coca_backbone(input_shape)
    x = base.output
    x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs=base.input, outputs=outputs, name='coca_classifier')

# InceptionV3 model
def create_inceptionv3_model(input_shape=(224, 224, 3), num_classes=len(DEFECT_CLASSES)):
    inputs = Input(shape=input_shape)
    x = Lambda(lambda x: x * 255.0)(inputs)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    base_model = InceptionV3(include_top=False, input_tensor=x, pooling='avg')
    x = base_model.output
    x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = Model(inputs, outputs, name='inceptionv3_classifier')
    return model

# Define models
MODELS = {
    'VisionTransformer': {'model_function': create_vit_model, 'preprocess_input': lambda x: x},
    'CoCa': {'model_function': create_coca_model, 'preprocess_input': lambda x: x},
    'YOLOv9': {'model_function': create_yolov9_model, 'preprocess_input': lambda x: x},
    'InceptionV3': {'model_function': create_inceptionv3_model, 'preprocess_input': lambda x: x}
}

# Custom tqdm callback
class TqdmCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, verbose=1):
        super(TqdmCallback, self).__init__()
        self.epochs = epochs
        self.verbose = verbose
        self.progbar = None

    def on_train_begin(self, logs=None):
        if self.verbose:
            self.progbar = tqdm(total=self.epochs, desc="Training", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose and self.progbar:
            logs = logs or {}
            self.progbar.set_postfix({
                'loss': f"{logs.get('loss', 0):.4f}",
                'acc': f"{logs.get('accuracy', 0):.4f}",
                'val_loss': f"{logs.get('val_loss', 0):.4f}",
                'val_acc': f"{logs.get('val_accuracy', 0):.4f}"
            })
            self.progbar.update(1)

    def on_train_end(self, logs=None):
        if self.verbose and self.progbar:
            self.progbar.close()

# Data generators
def create_data_generators(batch_size):
    logger.info(f"Creating data generators with batch_size={batch_size}...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    logger.info("Class Indices:")
    for class_name, index in train_generator.class_indices.items():
        logger.info(f"{index}: {class_name}")
    logger.info("Dataset Statistics:")
    logger.info(f"Training samples: {train_generator.samples}")
    logger.info(f"Validation samples: {validation_generator.samples}")
    logger.info(f"Test samples: {test_generator.samples}")
    logger.info(f"Number of classes: {len(train_generator.class_indices)}")
    return train_generator, validation_generator, test_generator

class GradientAccumulationModel(tf.keras.Model):
    def __init__(self, model, accumulation_steps=4):
        super().__init__()
        self.model = model
        self.accumulation_steps = tf.constant(accumulation_steps, dtype=tf.int64)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in model.trainable_variables]

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training, mask=mask)

    def compile(self, optimizer, loss, metrics, **kwargs):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs
        )
        self.optimizer = self.model.optimizer
        self.compiled_loss = self.model.compiled_loss
        self.compiled_metrics = self.model.compiled_metrics
        super().compile(
            optimizer=self.optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs
        )

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.model.losses)
            scaled_loss = loss / tf.cast(self.accumulation_steps, dtype=loss.dtype)
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
        
        current_iteration = self.optimizer.iterations.numpy() + 1
        if current_iteration % self.accumulation_steps.numpy() == 0:
            self._apply_accumulated_gradients()
        
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def _apply_accumulated_gradients(self):
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.model.trainable_variables))
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.model.trainable_variables[i], dtype=tf.float32))

    def test_step(self, data):
        return self.model.test_step(data)

    def predict_step(self, data):
        return self.model.predict_step(data)
    
    def count_params(self):
        return self.model.count_params()

# Create and compile model
def create_model(model_name):
    logger.info(f"Creating {model_name} model...")
    model_config = MODELS[model_name]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model = model_config['model_function'](input_shape=input_shape, num_classes=len(DEFECT_CLASSES))
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=METRICS
    )
    logger.info(f"{model_name} model created with {model.count_params():,} parameters")
    return model

# Callbacks
def create_callbacks(model_name, run_id):
    model_path = Path(MODEL_DIR) / f"{model_name}_{run_id}.h5"
    tensorboard_log_dir = Path(LOG_DIR) / f"{model_name}_{run_id}"
    tensorboard_log_dir.mkdir(exist_ok=True)
    callbacks = [
        ModelCheckpoint(str(model_path), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=str(tensorboard_log_dir), histogram_freq=0, write_graph=True, update_freq='epoch'),
        TqdmCallback(epochs=EPOCHS)
    ]
    logger.info(f"Created callbacks for {model_name}")
    return callbacks, str(model_path)

# Mixup augmentation
def mixup_data(x, y, alpha=0.2):
    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    x_mix = alpha * x + (1 - alpha) * tf.gather(x, indices)
    y_mix = alpha * y + (1 - alpha) * tf.gather(y, indices)
    return x_mix, y_mix

class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator, alpha=0.2):
        self.generator = generator
        self.alpha = alpha

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        x_batch, y_batch = self.generator[index]
        x_batch_mixed, y_batch_mixed = mixup_data(x_batch, y_batch, self.alpha)
        return x_batch_mixed, y_batch_mixed

# Evaluate model
def evaluate_model(model, test_generator, model_name, run_id, class_indices):
    logger.info(f"Evaluating {model_name} on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    logger.info(f"{model_name} Test accuracy: {test_accuracy:.4f}")
    logger.info(f"{model_name} Test loss: {test_loss:.4f}")
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    cm = confusion_matrix(y_true, y_pred_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_classes, average=None, labels=np.arange(len(class_names)), zero_division=0
    )
    cr = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True, zero_division=0)
    model_result_dir = Path(RESULTS_DIR) / f"{model_name}_{run_id}"
    model_result_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    cm_path = model_result_dir / 'confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    class_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    class_metrics.to_csv(model_result_dir / 'class_metrics.csv', index=False)
    report_df = pd.DataFrame(cr).transpose()
    report_df.to_csv(model_result_dir / 'classification_report.csv')
    return {
        'model_name': model_name,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'class_metrics': class_metrics,
        'class_names': class_names,
        'classification_report': cr,
        'confusion_matrix_path': str(cm_path)
    }

# Save history plots
def save_history_plots(history, model_name, run_id):
    model_result_dir = Path(RESULTS_DIR) / f"{model_name}_{run_id}"
    model_result_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    history_plot_path = model_result_dir / 'training_history.png'
    plt.savefig(history_plot_path, dpi=300)
    plt.close()
    history_df = pd.DataFrame(history.history)
    history_csv_path = model_result_dir / 'training_history.csv'
    history_df.to_csv(history_csv_path, index=False)
    return str(history_plot_path), str(history_csv_path)

# Train models
def train_models(train_generator, validation_generator, test_generator):
    logger.info("Starting multi-model training...")
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    all_metrics = {}

    for model_name in tqdm(MODELS.keys(), desc="Training Models", unit="model"):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")
        model_start_time = time.time()
        batch_size = model_batch_sizes.get(model_name, BATCH_SIZE)
        train_gen, val_gen, _ = create_data_generators(batch_size)
        model = create_model(model_name)
        callbacks, model_path = create_callbacks(model_name, run_id)
        x, y = next(train_gen)
        logger.info(f"Batch shape: {x.shape}, Labels shape: {y.shape}")
        logger.info(f"Starting initial training for {model_name}...")
        try:
            history = model.fit(
                train_gen,
                epochs=EPOCHS,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
            if TRANSFER_LEARNING:
                logger.info(f"Starting fine-tuning for {model_name}...")
                for layer in model.layers:
                    layer.trainable = True
                model.compile(
                    optimizer=Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
                    loss=LOSS,
                    metrics=METRICS
                )
                fine_tuning_callbacks = callbacks[:-1] + [TqdmCallback(epochs=FINE_TUNING_EPOCHS)]
                ft_history = model.fit(
                    train_gen,
                    epochs=FINE_TUNING_EPOCHS,
                    validation_data=val_gen,
                    callbacks=fine_tuning_callbacks,
                    verbose=1
                )
                for key in ft_history.history:
                    history.history[key].extend(ft_history.history[key])
            logger.info(f"Loading best weights from {model_path}")
            model.load_weights(model_path)
            history_plot_path, history_csv_path = save_history_plots(history, model_name, run_id)
            metrics = evaluate_model(model, test_generator, model_name, run_id, train_gen.class_indices)
            training_time = time.time() - model_start_time
            metrics['training_time'] = training_time
            metrics['history_plot_path'] = history_plot_path
            metrics['history_csv_path'] = history_csv_path
            metrics['model_path'] = model_path
            results.append({
                'model_name': model_name,
                'test_accuracy': metrics['test_accuracy'],
                'test_loss': metrics['test_loss'],
                'training_time': training_time,
                'best_val_accuracy': max(history.history['val_accuracy']),
                'best_val_loss': min(history.history['val_loss']),
                'weighted_f1': metrics['classification_report']['weighted avg']['f1-score'],
                'path': model_path
            })
            all_metrics[model_name] = metrics
            logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
            logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}, F1-Score (weighted): {metrics['classification_report']['weighted avg']['f1-score']:.4f}")
        finally:
            train_gen.reset()
            val_gen.reset()
            test_generator.reset()
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        logger.info(f"Memory cleared after training {model_name}")

    compare_models(results, all_metrics, train_generator.class_indices, run_id)
    tf.keras.backend.clear_session()
    gc.collect()

# Combined performance plot
def create_combined_performance_plot(results, run_id):
    df = pd.DataFrame(results)
    df = df.sort_values('test_accuracy', ascending=False)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(x='model_name', y='test_accuracy', data=df)
    plt.title('Test Accuracy')
    plt.ylim([0, 1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.subplot(2, 2, 2)
    sns.barplot(x='model_name', y='best_val_accuracy', data=df)
    plt.title('Best Validation Accuracy')
    plt.ylim([0, 1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.subplot(2, 2, 3)
    sns.barplot(x='model_name', y='weighted_f1', data=df)
    plt.title('Weighted F1-Score')
    plt.ylim([0, 1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.subplot(2, 2, 4)
    sns.barplot(x='model_name', y='training_time', data=df)
    plt.title('Training Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = Path(RESULTS_DIR) / f'combined_performance_{run_id}.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return str(plot_path)

# PDF report
def create_pdf_report(results, all_metrics, class_indices, run_id):
    sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    best_model = sorted_results[0]['model_name']
    pdf_path = Path(RESULTS_DIR) / f"perovskite_model_report_{run_id}.pdf"
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        plt.text(0.5, 0.7, "Perovskite Solar Cell\nDefect Classification Models", ha='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.5, f"Results Report\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha='center', fontsize=18)
        plt.text(0.5, 0.3, f"Best Model: {best_model}\nAccuracy: {sorted_results[0]['test_accuracy']:.4f}", ha='center', fontsize=16)
        plt.savefig(pdf, format='pdf')
        plt.close()
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        plt.text(0.5, 0.95, "Model Performance Summary", ha='center', fontsize=16, weight='bold')
        summary_text = "Model Comparison:\n\n"
        for i, result in enumerate(sorted_results):
            summary_text += f"{i+1}. {result['model_name']}:\n"
            summary_text += f"   - Test Accuracy: {result['test_accuracy']:.4f}\n"
            summary_text += f"   - Validation Accuracy: {result['best_val_accuracy']:.4f}\n"
            summary_text += f"   - F1-Score (weighted): {result['weighted_f1']:.4f}\n"
            summary_text += f"   - Training Time: {result['training_time']:.2f} seconds\n\n"
        plt.text(0.1, 0.85, summary_text, fontsize=12, va='top')
        plt.savefig(pdf, format='pdf')
        plt.close()
        plt.figure(figsize=(12, 10))
        img = plt.imread(Path(RESULTS_DIR) / f'combined_performance_{run_id}.png')
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(pdf, format='pdf')
        plt.close()
        for model_name in MODELS.keys():
            metrics = all_metrics[model_name]
            plt.figure(figsize=(12, 2))
            plt.axis('off')
            plt.text(0.5, 0.5, f"{model_name} Detailed Results", ha='center', fontsize=16, weight='bold')
            plt.savefig(pdf, format='pdf')
            plt.close()
            plt.figure(figsize=(12, 6))
            img = plt.imread(metrics['history_plot_path'])
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(pdf, format='pdf')
            plt.close()
            plt.figure(figsize=(12, 10))
            img = plt.imread(metrics['confusion_matrix_path'])
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(pdf, format='pdf')
            plt.close()
            plt.figure(figsize=(12, 6))
            plt.axis('off')
            plt.text(0.5, 0.95, f"{model_name} - Per-Class Metrics", ha='center', fontsize=14, weight='bold')
            class_metrics_text = "Class\t\tPrecision\tRecall\t\tF1-Score\tSupport\n" + "-"*80 + "\n"
            for i, row in metrics['class_metrics'].iterrows():
                class_metrics_text += f"{row['Class']}\t\t{row['Precision']:.4f}\t\t{row['Recall']:.4f}\t\t{row['F1-Score']:.4f}\t\t{int(row['Support'])}\n"
            class_metrics_text += "-"*80 + "\n"
            weighted_metrics = metrics['classification_report']['weighted avg']
            class_metrics_text += f"Weighted Avg\t{weighted_metrics['precision']:.4f}\t\t{weighted_metrics['recall']:.4f}\t\t{weighted_metrics['f1-score']:.4f}\t\t{int(weighted_metrics['support'])}"
            plt.text(0.1, 0.85, class_metrics_text, fontsize=10, va='top', family='monospace')
            plt.savefig(pdf, format='pdf')
            plt.close()
    logger.info(f"PDF report generated: {pdf_path}")
    return str(pdf_path)

# Compare models
def compare_models(results, all_metrics, class_indices, run_id):
    logger.info("Comparing model performance...")
    combined_plot_path = create_combined_performance_plot(results, run_id)
    df = pd.DataFrame(results)
    df = df.sort_values('test_accuracy', ascending=False)
    detailed_results = []
    for model_name in MODELS.keys():
        metrics = all_metrics[model_name]
        class_metrics = metrics['class_metrics']
        row = {
            'model_name': model_name,
            'test_accuracy': metrics['test_accuracy'],
            'test_loss': metrics['test_loss'],
            'training_time_seconds': metrics['training_time'],
            'weighted_precision': metrics['classification_report']['weighted avg']['precision'],
            'weighted_recall': metrics['classification_report']['weighted avg']['recall'],
            'weighted_f1': metrics['classification_report']['weighted avg']['f1-score'],
            'model_path': metrics['model_path']
        }
        for i, class_name in enumerate(metrics['class_names']):
            row[f"{class_name}_precision"] = metrics['precision'][i]
            row[f"{class_name}_recall"] = metrics['recall'][i]
            row[f"{class_name}_f1"] = metrics['f1'][i]
            row[f"{class_name}_support"] = metrics['classification_report'][class_name]['support']
        detailed_results.append(row)
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df = detailed_df.sort_values('test_accuracy', ascending=False)
    summary_path = Path(RESULTS_DIR) / f'model_comparison_summary_{run_id}.csv'
    df.to_csv(summary_path, index=False)
    detailed_path = Path(RESULTS_DIR) / f'model_comparison_detailed_{run_id}.csv'
    detailed_df.to_csv(detailed_path, index=False)
    pdf_path = create_pdf_report(results, all_metrics, class_indices, run_id)
    logger.info(f"\n{'='*50}")
    logger.info("MODEL COMPARISON RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"\nDetailed results saved to: {detailed_path}")
    logger.info(f"Summary results saved to: {summary_path}")
    logger.info(f"Combined performance plot: {combined_plot_path}")
    logger.info(f"PDF report: {pdf_path}")
    logger.info(f"\nTop performing models:")
    top_n = min(3, len(df))
    for i in range(top_n):
        row = df.iloc[i]
        logger.info(f"{i+1}. {row['model_name']}: Accuracy = {row['test_accuracy']:.4f}, F1 = {row['weighted_f1']:.4f}")
    logger.info(f"{'='*50}")
    logger.info(f"Best model: {df.iloc[0]['model_name']}")
    logger.info(f"Model saved at: {df.iloc[0]['path']}")
    logger.info(f"{'='*50}")

# Verify dataset paths
def verify_dataset_paths():
    required_dirs = [("Training", TRAIN_DIR), ("Validation", VALIDATION_DIR), ("Testing", TEST_DIR)]
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    for name, dir_path in required_dirs:
        if not Path(dir_path).exists():
            logger.error(f"{name} directory not found: {dir_path}")
            return False
        subdirs = [d for d in os.listdir(dir_path) if Path(dir_path, d).is_dir()]
        if not subdirs:
            logger.error(f"No class subdirectories found in {name} directory: {dir_path}")
            return False
        for subdir in subdirs:
            full_path = Path(dir_path) / subdir
            files = [f for f in os.listdir(full_path) if Path(full_path, f).is_file()
                     and f.lower().endswith(supported_extensions)]
            if not files:
                logger.error(f"No image files found in {name}/{subdir}")
                return False
            logger.info(f"Found {len(files)} images in {name}/{subdir} with extensions: {sorted(set(os.path.splitext(f)[1].lower() for f in files))}")
    logger.info("All dataset directories verified successfully")
    return True

# Create visualization directory
def create_visualization_dir():
    Path(VISUALIZATION_DIR).mkdir(exist_ok=True)
    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    logger.info(f"Created directories: {VISUALIZATION_DIR}, {MODEL_DIR}, {RESULTS_DIR}")

# Custom JSON encoder
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tf.Variable):
            return obj.numpy().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomJSONEncoder, self).default(obj)

# Save model info
def save_model_info(model_info, run_id):
    info = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "parameters": {
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "fine_tuning_epochs": FINE_TUNING_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "fine_tuning_learning_rate": FINE_TUNING_LEARNING_RATE,
            "transfer_learning": TRANSFER_LEARNING,
            "defect_classes": DEFECT_CLASSES
        },
        "models": list(MODELS.keys())
    }
    info_path = Path(RESULTS_DIR) / f"training_info_{run_id}.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4, cls=CustomJSONEncoder)
    logger.info(f"Training parameters saved to {info_path}")
    return str(info_path)

# Main function
def main():
    start_time = time.time()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        except RuntimeError as e:
            logger.error(f"Failed to set memory growth: {e}")
    create_visualization_dir()
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"Starting training run {run_id}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    logger.info(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")
    save_model_info(MODELS, run_id)
    if not verify_dataset_paths():
        logger.error("Dataset verification failed. Exiting.")
        return
    try:
        train_generator, validation_generator, test_generator = create_data_generators(BATCH_SIZE)
        train_models(train_generator, validation_generator, test_generator)
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        train_generator.reset()
        validation_generator.reset()
        test_generator.reset()
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Multi-model training completed successfully.")

if __name__ == "__main__":
    main()
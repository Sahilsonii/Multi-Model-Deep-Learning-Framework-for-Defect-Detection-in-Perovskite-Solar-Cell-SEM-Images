import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50V2,
    EfficientNetB3, 
    DenseNet169,
    MobileNetV3Large
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
from datetime import datetime
import logging
import json
from tqdm import tqdm  # Correct import for command-line use
import time
from matplotlib.backends.backend_pdf import PdfPages
import sys
from pathlib import Path

# Import configurations
from config import (
    IMAGE_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, DEFECT_CLASSES, MODEL_DIR, EPOCHS, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, TRANSFER_LEARNING, FINE_TUNING_EPOCHS,
    FINE_TUNING_LEARNING_RATE, LOSS, METRICS, TRAIN_DIR, VALIDATION_DIR,
    TEST_DIR, LEARNING_RATE, VISUALIZATION_DIR
)

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the models to train
MODELS = {
    'ResNet50V2': {
        'model_function': ResNet50V2,
        'weights': 'imagenet',
        'preprocess_input': tf.keras.applications.resnet_v2.preprocess_input
    },
    'EfficientNetB3': {
        'model_function': EfficientNetB3,
        'weights': 'imagenet',
        'preprocess_input': tf.keras.applications.efficientnet.preprocess_input
    },
    'DenseNet169': {
        'model_function': DenseNet169,
        'weights': 'imagenet',
        'preprocess_input': tf.keras.applications.densenet.preprocess_input
    },
    'MobileNetV3Large': {
        'model_function': MobileNetV3Large,
        'weights': 'imagenet',
        'preprocess_input': tf.keras.applications.mobilenet_v3.preprocess_input
    }
}

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

def create_data_generators():
    """Create data generators for training, validation, and testing."""
    
    logger.info("Creating data generators...")
    
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
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

def create_model(model_name):
    """Create and compile a model based on the specified architecture."""
    
    logger.info(f"Creating {model_name} model...")
    
    model_config = MODELS[model_name]
    base_model = model_config['model_function'](
        weights=model_config['weights'],
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(DEFECT_CLASSES), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if TRANSFER_LEARNING:
        for layer in base_model.layers:
            layer.trainable = False
        logger.info(f"Base model layers frozen for transfer learning")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=METRICS
    )
    
    logger.info(f"{model_name} model created with {model.count_params():,} parameters")
    
    return model, base_model

def create_callbacks(model_name, run_id):
    """Create training callbacks for a specific model."""
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{run_id}.h5")
    tensorboard_log_dir = os.path.join(LOG_DIR, f"{model_name}_{run_id}")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        TqdmCallback(epochs=EPOCHS)
    ]
    
    logger.info(f"Created callbacks for {model_name}: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard")
    
    return callbacks, model_path

def evaluate_model(model, test_generator, model_name, run_id, class_indices):
    """Evaluate the model and generate performance metrics."""
    
    logger.info(f"Evaluating {model_name} on test set...")
    
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    logger.info(f"{model_name} Test accuracy: {test_accuracy:.4f}")
    logger.info(f"{model_name} Test loss: {test_loss:.4f}")
    
    logger.info(f"Generating predictions for {model_name}...")
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    y_true = test_generator.classes
    
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    
    cm = confusion_matrix(y_true, y_pred_classes)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_classes, average=None, labels=np.arange(len(class_names))
    )
    
    cr = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    
    model_result_dir = os.path.join(RESULTS_DIR, f"{model_name}_{run_id}")
    os.makedirs(model_result_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(model_result_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    class_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    class_metrics.to_csv(os.path.join(model_result_dir, 'class_metrics.csv'), index=False)
    
    report_df = pd.DataFrame(cr).transpose()
    report_df.to_csv(os.path.join(model_result_dir, 'classification_report.csv'))
    
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
        'confusion_matrix_path': cm_path
    }

def save_history_plots(history, model_name, run_id):
    """Save training and validation accuracy/loss plots."""
    
    model_result_dir = os.path.join(RESULTS_DIR, f"{model_name}_{run_id}")
    os.makedirs(model_result_dir, exist_ok=True)
    
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
    history_plot_path = os.path.join(model_result_dir, 'training_history.png')
    plt.savefig(history_plot_path, dpi=300)
    plt.close()
    
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(model_result_dir, 'training_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    
    return history_plot_path, history_csv_path

def train_models(train_generator, validation_generator, test_generator):
    """Train multiple models on the same dataset and compare results."""
    
    logger.info("Starting multi-model training...")
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = []
    all_metrics = {}
    
    for model_name in tqdm(MODELS.keys(), desc="Training Models", unit="model"):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")
        
        model_start_time = time.time()
        
        model, base_model = create_model(model_name)
        callbacks, model_path = create_callbacks(model_name, run_id)
        
        logger.info(f"Starting initial training for {model_name}...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=0
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
                train_generator,
                epochs=FINE_TUNING_EPOCHS,
                validation_data=validation_generator,
                callbacks=fine_tuning_callbacks,
                verbose=0
            )
            
            for key in ft_history.history:
                history.history[key].extend(ft_history.history[key])
        
        logger.info(f"Loading best weights from {model_path}")
        model.load_weights(model_path)
        
        history_plot_path, history_csv_path = save_history_plots(history, model_name, run_id)
        metrics = evaluate_model(model, test_generator, model_name, run_id, train_generator.class_indices)
        
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
        
        tf.keras.backend.clear_session()
    
    compare_models(results, all_metrics, train_generator.class_indices, run_id)

def create_combined_performance_plot(results):
    """Create a combined performance plot for all models."""
    
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
    plot_path = os.path.join(RESULTS_DIR, 'combined_performance.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    return plot_path

def create_pdf_report(results, all_metrics, class_indices, run_id):
    """Create a comprehensive PDF report with all results and visualizations."""
    
    sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    best_model = sorted_results[0]['model_name']
    
    pdf_path = os.path.join(RESULTS_DIR, f"perovskite_model_report_{run_id}.pdf")
    
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        plt.text(0.5, 0.7, "Perovskite Solar Cell\nDefect Classification Models", 
                 ha='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.5, f"Results Report\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                 ha='center', fontsize=18)
        plt.text(0.5, 0.3, f"Best Model: {best_model}\nAccuracy: {sorted_results[0]['test_accuracy']:.4f}", 
                 ha='center', fontsize=16)
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
        img = plt.imread(os.path.join(RESULTS_DIR, 'combined_performance.png'))
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
    return pdf_path

def compare_models(results, all_metrics, class_indices, run_id):
    """Compare the performance of all trained models."""
    
    logger.info("Comparing model performance...")
    
    combined_plot_path = create_combined_performance_plot(results)
    
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
    
    summary_path = os.path.join(RESULTS_DIR, f'model_comparison_summary_{run_id}.csv')
    df.to_csv(summary_path, index=False)
    
    detailed_path = os.path.join(RESULTS_DIR, f'model_comparison_detailed_{run_id}.csv')
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

def verify_dataset_paths():
    """Verify that all dataset directories exist and contain data."""
    
    dataset_path = r"C:\Users\ASUS\Desktop\New folder (2)\dataset"
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return False
    
    global TRAIN_DIR, VALIDATION_DIR, TEST_DIR
    TRAIN_DIR = str(dataset_path / 'train')
    VALIDATION_DIR = str(dataset_path / 'validation')
    TEST_DIR = str(dataset_path / 'test')
    
    required_dirs = [
        ("Training", TRAIN_DIR),
        ("Validation", VALIDATION_DIR),
        ("Testing", TEST_DIR)
    ]
    
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    
    for name, dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.error(f"{name} directory not found: {dir_path}")
            return False
        
        subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        if not subdirs:
            logger.error(f"No class subdirectories found in {name} directory: {dir_path}")
            return False
        
        for subdir in subdirs:
            full_path = os.path.join(dir_path, subdir)
            files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f)) 
                    and f.lower().endswith(supported_extensions)]
            if not files:
                logger.error(f"No image files found in {name}/{subdir}")
                return False
            logger.info(f"Found {len(files)} images in {name}/{subdir} with extensions: {sorted(set(os.path.splitext(f)[1].lower() for f in files))}")
    
    logger.info("All dataset directories verified successfully")
    return True

def create_visualization_dir():
    """Create visualization directory for storing result plots."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    logger.info(f"Created visualization directory: {VISUALIZATION_DIR}")

def save_model_info(model_info, run_id):
    """Save model architecture and training parameters for reproducibility."""
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
    
    info_path = os.path.join(RESULTS_DIR, f"training_info_{run_id}.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    logger.info(f"Training parameters saved to {info_path}")
    return info_path

def main():
    """Main function to run the multi-model training and evaluation pipeline."""
    
    start_time = time.time()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
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
        train_generator, validation_generator, test_generator = create_data_generators()
    except Exception as e:
        logger.error(f"Error creating data generators: {str(e)}")
        return
    
    try:
        train_models(train_generator, validation_generator, test_generator)
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Multi-model training completed successfully.")

if __name__ == "__main__":
    main()
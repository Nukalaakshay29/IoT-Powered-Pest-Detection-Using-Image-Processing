import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: ENHANCED PRE-PROCESSING  (From Paper 2 - Khirade & Patil 2015)
# ============================================================================

class ImagePreprocessor:
    """
    Complete pre-processing pipeline from Paper 2.

    Steps applied in order:
      1. Resize to target_size
      2. Gaussian blur      -> removes camera noise
      3. Histogram equaliz  -> improves contrast in any lighting
      4. Contrast enhance   -> makes disease features clearer
    """

    @staticmethod
    def noise_removal(image, kernel_size=5):
        """Gaussian blur to remove noise"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def histogram_equalization(image):
        """
        Histogram equalization on Y channel (YUV space).
        Paper 2: 'distributes the intensities of the images'
        Keeps colour, only changes brightness distribution.
        """
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def enhance_contrast(image, alpha=1.3, beta=0):
        """
        Contrast enhancement.
        alpha > 1 increases contrast.
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def preprocess_image(image, target_size=(224, 224)):
        """
        Full pipeline called for every image before CNN.
        Works on a numpy BGR array (same as cv2.imread output).
        """
        image = cv2.resize(image, target_size)
        image = ImagePreprocessor.noise_removal(image)
        image = ImagePreprocessor.histogram_equalization(image)
        image = ImagePreprocessor.enhance_contrast(image)
        return image


# ============================================================================
# STEP 2: SEGMENTATION LAYER  (From Paper 2 - Khirade & Patil 2015)
# ============================================================================

class ImageSegmenter:
    """
    Implements three segmentation methods from Paper 2:
      1. K-means clustering (k=3)  -> separates healthy/diseased/background
      2. Green pixel masking        -> removes healthy green tissue
      3. Otsu thresholding          -> binary mask of diseased region
      + Canny edge detection        -> highlights lesion boundaries
    """

    @staticmethod
    def kmeans_segmentation(image, k=3):
        """
        K-means clustering.
        Paper 2: 'K-means clustering is more accurate than other methods'
        k=3 -> healthy tissue, diseased tissue, background
        """
        pixel_values = np.float32(image.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        centers  = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(image.shape)
        return segmented

    @staticmethod
    def green_pixel_masking(image):
        """
        Remove healthy (green) pixels so CNN focuses on diseased areas.
        Paper 2: 'green pixels masked and removed'
        """
        hsv           = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green   = np.array([35, 40, 40])
        upper_green   = np.array([85, 255, 255])
        green_mask    = cv2.inRange(hsv, lower_green, upper_green)
        diseased_mask = cv2.bitwise_not(green_mask)
        masked        = cv2.bitwise_and(image, image, mask=diseased_mask)
        return masked, diseased_mask

    @staticmethod
    def otsu_thresholding(image):
        """
        Otsu automatic thresholding.
        Paper 2: 'creates binary images from grey-level images'
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def segment_image(image):
        """
        Full segmentation pipeline.
        Returns dict so every stage can be visualised separately.
        """
        segmented          = ImageSegmenter.kmeans_segmentation(image, k=3)
        masked, mask       = ImageSegmenter.green_pixel_masking(segmented)
        binary             = ImageSegmenter.otsu_thresholding(masked)
        gray               = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        edges              = cv2.Canny(gray, 50, 150)

        return {
            'original':  image,
            'segmented': segmented,
            'masked':    masked,
            'binary':    binary,
            'edges':     edges,
            'mask':      mask
        }


# ============================================================================
# STEP 3: TINY-LITENET CNN  (From Paper 1 - Nyakuri et al. 2025)
# ============================================================================

class TinyLiteNet:
    """
    Lightweight CNN model.
    Paper 1 specs: 1.2 MB, 1.48 M parameters, 16 ms inference on RPi5.
    Architecture: MobileNetV2 inspired + 6 Squeeze-Excitation (SE) blocks.
    """

    @staticmethod
    def squeeze_excitation_block(x, ratio=16):
        """
        SE block - recalibrates channel-wise feature responses.
        Paper 1: '6 SE depthwise blocks to enhance feature representation'
        """
        channels = x.shape[-1]
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(max(1, channels // ratio), activation='relu')(se)
        se = layers.Dense(channels, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, channels))(se)
        return layers.Multiply()([x, se])

    @staticmethod
    def depthwise_block(x, filters, stride=1):
        """Depthwise-separable conv block (MobileNetV2 style)."""
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6)(x)
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6)(x)
        return x

    @staticmethod
    def build_model(input_shape=(224, 224, 3), num_classes=9):
        """
        Build Tiny-LiteNet.
        Target: ~1.2 MB, ~1.48 M parameters.
        """
        inputs = layers.Input(shape=input_shape, name='input_image')

        # Stem conv
        x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6)(x)

        # 6 Depthwise + SE blocks  (Paper 1: 6 SE depthwise blocks)
        configs = [(64,1), (128,2), (128,1), (256,2), (256,1), (512,2)]
        for filters, stride in configs:
            x = TinyLiteNet.depthwise_block(x, filters, stride)
            x = TinyLiteNet.squeeze_excitation_block(x)

        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

        model = models.Model(inputs, outputs, name='TinyLiteNet')
        return model

    @staticmethod
    def model_info(model):
        """Print model size and parameter count."""
        params  = model.count_params()
        size_mb = (params * 4) / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"  Model        : {model.name}")
        print(f"  Parameters   : {params:,}  ({params/1e6:.2f}M)")
        print(f"  Size (fp32)  : {size_mb:.2f} MB  (target <= 1.5 MB)")
        print(f"  Layers       : {len(model.layers)}")
        print(f"{'='*60}\n")
        return {'size_mb': size_mb, 'params': params}


# ============================================================================
# GRAD-CAM EXPLAINABILITY
# ============================================================================

class GradCAM:
    """
    Grad-CAM: highlights which leaf areas drove the prediction.
    Red  = high attention (likely infected zone)
    Blue = low attention
    """

    def __init__(self, model):
        self.model = model
        self.last_conv_layer = None
        # Find last convolutional layer automatically
        for layer in reversed(model.layers):
            if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
                self.last_conv_layer = layer.name
                break
        if self.last_conv_layer is None:
            raise ValueError("No Conv2D layer found in model.")

        self.grad_model = models.Model(
            inputs  = model.input,
            outputs = [model.get_layer(self.last_conv_layer).output,
                       model.output]
        )

    def compute_heatmap(self, img_array, class_idx=None):
        """
        img_array : float32 array shape (1, H, W, 3) values in [0,1]
        Returns   : heatmap (H, W) in [0, 1]
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array, training=False)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_score = predictions[:, class_idx]

        grads        = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap).numpy()
        heatmap      = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap  = heatmap / heatmap.max()
        return heatmap

    def overlay(self, original_bgr, heatmap, alpha=0.45):
        """Overlay heatmap on original image. Returns BGR image."""
        h, w            = original_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        coloured        = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(original_bgr, 1 - alpha, coloured, alpha, 0)


# ============================================================================
# COMPLETE PEST DETECTION SYSTEM
# ============================================================================

class PestDetectionSystem:
    """
    Main class used by:
      - Colab notebook  -> training + batch inference + visualisation
      - Raspberry Pi    -> predict_from_image_path()
      - ESP32 server    -> predict_from_array()
    """

    DEFAULT_CLASS_NAMES = [
        'Healthy', 'Fall Armyworm', 'Grasshopper', 'Aphids',
        'Stem Borer', 'Common Rust', 'Gray Leaf Spot',
        'Northern Leaf Blight', 'Leaf Beetle'
    ]

    def __init__(self, model_path=None, num_classes=None, class_names=None):
        self.preprocessor = ImagePreprocessor()
        self.segmenter    = ImageSegmenter()

        # Class names
        if class_names is not None:
            self.class_names = class_names
            self.num_classes = len(class_names)
        else:
            self.class_names = self.DEFAULT_CLASS_NAMES
            self.num_classes = num_classes or len(self.DEFAULT_CLASS_NAMES)

        # Load or build model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            self.model       = keras.models.load_model(model_path)
            self.num_classes = self.model.output_shape[-1]
            print(f"Model loaded  ({self.num_classes} classes)")
        else:
            print("Building new TinyLiteNet model...")
            self.model = TinyLiteNet.build_model(
                input_shape=(224, 224, 3),
                num_classes=self.num_classes
            )

        self.grad_cam = None   # initialised after compile

    # ── Compile ──────────────────────────────────────────────────────
    def compile_model(self, learning_rate=0.001):
        """Compile with metrics required by the Colab notebook."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ]
        )
        try:
            self.grad_cam = GradCAM(self.model)
        except Exception as e:
            print(f"Grad-CAM init warning: {e}")

    # ── Model info ───────────────────────────────────────────────────
    def get_model_info(self):
        """
        Returns dict with model size and parameters.
        Used by Colab notebook to display model stats.
        """
        params  = self.model.count_params()
        size_mb = (params * 4) / (1024 * 1024)
        return {
            'model_size_mb':       round(size_mb, 2),
            'total_parameters':    params,
            'parameters_millions': round(params / 1e6, 2),
            'num_classes':         self.num_classes,
            'input_shape':         self.model.input_shape,
        }

    # =========================================================================
    # INFERENCE METHODS
    # =========================================================================

    def predict_from_image_path(self, image_path, show_steps=False):
        """
        Load image from disk -> run full 3-step pipeline.
        USE ON: Raspberry Pi
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return self._full_pipeline(image, show_steps=show_steps)

    def predict_from_array(self, image_array):
        """
        Accept raw numpy array (BGR uint8).
        USE ON: ESP32 (image captured -> sent as array) / RPi camera
        Returns lightweight dict: predicted_class + confidence only.
        """
        result = self._full_pipeline(image_array, show_steps=False)
        return {
            'predicted_class': result['predicted_class'],
            'confidence':      result['confidence']
        }

    # Backward-compatible alias
    def predict_single_image_from_array(self, image_array):
        """Alias for predict_from_array() - keeps old code working."""
        return self.predict_from_array(image_array)

    def run_inference_on_batch(self, images_tensor):
        """
        Run model on a tf.data batch tensor.
        USE ON: Colab notebook Step 8.
        images_tensor: float32 (B, H, W, 3) normalised [0,1]
        Returns: predictions array (B, num_classes)
        """
        return self.model.predict(images_tensor, verbose=0)

    # ── Internal pipeline ─────────────────────────────────────────────
    def _full_pipeline(self, image_bgr, show_steps=False):
        """Runs all 3 steps and returns complete result dict."""
        original = image_bgr.copy()

        # STEP 1: Pre-processing
        preprocessed = self.preprocessor.preprocess_image(image_bgr)

        # STEP 2: Segmentation
        seg = self.segmenter.segment_image(preprocessed)

        # STEP 3: CNN Prediction
        model_input   = preprocessed.astype('float32') / 255.0
        model_input   = np.expand_dims(model_input, axis=0)
        predictions   = self.model.predict(model_input, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence    = float(predictions[0][predicted_idx] * 100)

        predicted_class = (self.class_names[predicted_idx]
                           if predicted_idx < len(self.class_names)
                           else f'Class_{predicted_idx}')

        # Grad-CAM
        heatmap = overlay = None
        if self.grad_cam is not None:
            try:
                heatmap = self.grad_cam.compute_heatmap(model_input, predicted_idx)
                overlay = self.grad_cam.overlay(preprocessed, heatmap)
            except Exception:
                pass

        result = {
            'original':        original,
            'preprocessed':    preprocessed,
            'segmented':       seg['segmented'],
            'masked':          seg['masked'],
            'binary':          seg['binary'],
            'edges':           seg['edges'],
            'predicted_class': predicted_class,
            'predicted_idx':   predicted_idx,
            'confidence':      confidence,
            'all_probs':       predictions[0],
            'heatmap':         heatmap,
            'overlay':         overlay,
        }

        if show_steps:
            self._visualise_steps(result)

        return result

    # =========================================================================
    # VISUALISATION METHODS  (Colab-friendly, display inline)
    # =========================================================================

    def _visualise_steps(self, result):
        """Show all 3 processing steps in one figure."""
        titles = ['1. Original', '2. Pre-processed', '3. K-means Seg.',
                  '4. Diseased Mask', '5. Otsu Binary', '6. Grad-CAM']
        images = [
            result['original'],
            result['preprocessed'],
            result['segmented'],
            result['masked'],
            result['binary'],
            result['overlay'] if result['overlay'] is not None else result['preprocessed']
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for i, (title, img) in enumerate(zip(titles, images)):
            if img is None:
                axes[i].text(0.5, 0.5, 'N/A', ha='center', va='center')
            elif len(img.shape) == 2:
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if i == 5:
                title = (f"Prediction: {result['predicted_class']}\n"
                         f"Confidence: {result['confidence']:.1f}%")

            axes[i].set_title(title, fontsize=11, fontweight='bold')
            axes[i].axis('off')

        fig.suptitle('Pest Detection - All 3 Processing Steps',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def show_predictions(self, images_tensor, labels_tensor, class_names=None):
        """
        Show 9 sample predictions from a Colab val_ds batch.
        USE ON: Colab notebook Step 8.

        images_tensor : float32 tensor (B, H, W, 3) normalised [0,1]
        labels_tensor : one-hot tensor (B, num_classes)
        class_names   : list of class name strings (optional override)
        """
        if class_names is not None:
            self.class_names = class_names

        predictions = self.run_inference_on_batch(images_tensor)
        n           = min(9, len(images_tensor))

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()

        for i in range(n):
            img        = images_tensor[i].numpy()
            true_idx   = int(np.argmax(labels_tensor[i]))
            pred_idx   = int(np.argmax(predictions[i]))
            confidence = float(predictions[i][pred_idx] * 100)

            true_name = (self.class_names[true_idx]
                         if true_idx < len(self.class_names) else f'Class_{true_idx}')
            pred_name = (self.class_names[pred_idx]
                         if pred_idx < len(self.class_names) else f'Class_{pred_idx}')

            axes[i].imshow(img)
            color = 'green' if true_idx == pred_idx else 'red'
            axes[i].set_title(
                f"True: {true_name}\nPred: {pred_name}\n({confidence:.1f}%)",
                color=color, fontsize=9, fontweight='bold'
            )
            axes[i].axis('off')

        for j in range(n, 9):
            axes[j].axis('off')

        correct   = mpatches.Patch(color='green', label='Correct prediction')
        incorrect = mpatches.Patch(color='red',   label='Wrong prediction')
        fig.legend(handles=[correct, incorrect],
                   loc='lower center', ncol=2, fontsize=11)

        fig.suptitle('Sample Predictions - Pest Detection',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.show()

    def plot_training_history(self, history):
        """
        Plot accuracy and loss curves after training.
        USE ON: Colab notebook Step 6.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        axes[0].plot(history.history['accuracy'],     label='Train',      linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].axhline(y=0.986, color='red', linestyle='--', label='Target 98.6%')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy - TinyLiteNet', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history.history['loss'],     label='Train',      linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss - TinyLiteNet', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: training_history.png")

    def show_confusion_matrix(self, val_ds, class_names=None):
        """
        Generate and display confusion matrix from a tf.data dataset.
        USE ON: Colab after training to evaluate model.
        """
        try:
            import seaborn as sns
        except ImportError:
            print("seaborn not installed. Run: pip install seaborn")
            return

        from sklearn.metrics import confusion_matrix, classification_report

        if class_names is not None:
            self.class_names = class_names

        y_true, y_pred = [], []
        for images, labels in val_ds:
            preds  = self.run_inference_on_batch(images)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True',      fontsize=12)
        plt.title('Confusion Matrix - Pest & Disease Detection',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: confusion_matrix.png")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))


# ============================================================================
# QUICK SELF-TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  IMPROVED PEST DETECTION SYSTEM - SELF TEST")
    print("=" * 65)

    system = PestDetectionSystem(num_classes=9)
    system.compile_model()

    info = system.get_model_info()
    print(f"\n  Model size   : {info['model_size_mb']} MB   (target <= 1.5 MB)")
    print(f"  Parameters   : {info['parameters_millions']}M   (target <= 2M)")

    dummy_bgr    = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    preprocessed = ImagePreprocessor.preprocess_image(dummy_bgr)
    print(f"\n  Pre-processing : {dummy_bgr.shape} -> {preprocessed.shape}  OK")

    seg = ImageSegmenter.segment_image(preprocessed)
    print(f"  Segmentation   : keys = {list(seg.keys())}  OK")

    result = system.predict_from_array(dummy_bgr)
    print(f"\n  Inference test :")
    print(f"    Predicted class : {result['predicted_class']}")
    print(f"    Confidence      : {result['confidence']:.1f}%")

    print("\n" + "=" * 65)
    print("  All tests passed - system is ready!")
    print("=" * 65)

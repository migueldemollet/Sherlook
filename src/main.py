import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.cm as cm
import matplotlib.pyplot as plt


IMG_SIZE = (256, 256)

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.model = load_model("./model/ela_models/detect_manipulated_images_model_EfficientNetB1.h5")

    def call(self, inputs):
        return self.model(inputs)
    
    def predict(self, image, grad_cam=False):
        img_preprocessed = self._preprocess_image(image)
        if grad_cam:
            return self._predict_with_grad_cam(image, img_preprocessed)
        else:
            return super(CustomModel, self).predict(img_preprocessed)
        
    def _preprocess_image(self, img):
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ela_image = ela(img)
        img_expanded = np.expand_dims(ela_image, axis=0)  
        return img_expanded

    def _get_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None
    
    def _predict_with_grad_cam(self, image, preprocess_image):
        image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self._get_last_conv_layer()).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(preprocess_image)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on the original image
        superimposed_img = jet_heatmap * 0.4 + image
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        return class_channel[0].numpy(), np.array(superimposed_img)
    
    @classmethod
    def load_model(cls, filepath):
        # Load the model architecture and weights
        with open(filepath + '/model_architecture.json', 'r') as json_file:
            model_json = json_file.read()
        loaded_model = cls._from_json(model_json)

        loaded_model.model.load_weights(filepath + '/model_weights.h5')
        return loaded_model

    @classmethod
    def _from_json(cls, model_json):
        # Create a model instance from JSON architecture
        model = cls()
        model.model = tf.keras.models.model_from_json(model_json)
        return model
    


def ela(image, quality=99):
    # Comprimir y descomprimir la imagen
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed_image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)

    diff = 15 * cv2.absdiff(image, compressed_image)
    
    return diff

def process_prediction(prediction):
    class_result = 0
    confidence = 0

    if prediction < 0.5:
        class_result = "Original"
        confidence = (1 - prediction) * 100
    else:
        class_result = "Modified"
        confidence = prediction * 100

    return class_result, confidence


if __name__ == "__main__":
    model = CustomModel()
    img_path = './dataset/test/me_x_3.jpg' # feel free to change the image path
    image = cv2.imread(img_path)

    pred, grad_cam = model.predict(image, grad_cam=True) # grad_cam=True to get the grad_cam image
    class_result, confidence = process_prediction(pred)

    print(f"Class: {class_result} - Confidence: {confidence}")
    plt.imshow(grad_cam)
    plt.show()

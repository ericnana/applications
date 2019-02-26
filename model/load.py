from keras.models import model_from_json
import tensorflow as tf
import keras
from keras.optimizers import Adam



def init():
    json_file = open('larger_model_multilayer_perceptron_cnn.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    #loaded_model.load_weights("simple_model_multilayer_perceptron_cnn.h5")
    loaded_model.load_weights("larger_model_multilayer_perceptron_cnn.h5")
    print("Loaded Model from disk")
    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer= Adam(lr = 0.01),metrics=['accuracy'])
    graph = tf.get_default_graph()
    return loaded_model,graph

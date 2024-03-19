import numpy as np
import pandas as pd
from glob import glob
import re
import scipy
import os
import sklearn
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer
import json
import pickle
import keras
import tensorflow as tf
from keras.models import load_model

#NN Function
def NN_func(X_train, X_test, Y_train, Y_test, label_classes, report_path, num_epochs=1500):
    tf.random.set_seed(66)
    test_name = "NN_" + str(X_train.shape[0]) + "samples_" + str(X_train.shape[1]) + "Hz"
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    #generate sample weights
    y_integers = np.argmax(Y_train, axis=1)
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_integers), y=y_integers)
    sample_weights = np.matmul(Y_train, class_weights)
    
    def make_model(input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same")(input_layer)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same")(conv2)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        deep = keras.layers.Dense(1500, activation="relu")(gap)

        #output layer
        output_layer = keras.layers.Dense(len(label_classes), activation="softmax")(deep)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    input_shape = X_train.shape[1:]
    NN_model = make_model(input_shape=input_shape)
    NN_model.compile(
                    loss = tf.keras.losses.CategoricalCrossentropy(),
                    optimizer = tf.keras.optimizers.Adam(),
                    metrics = ["categorical_accuracy"])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.00005)
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, min_delta=0.001)
    checkpoints = keras.callbacks.ModelCheckpoint((report_path + test_name + "_model.h5"), 
                                                    save_best_only=True, save_freq="epoch")

    history = NN_model.fit(
        X_train, Y_train, epochs=num_epochs, #max number of epochs if val_loss doesn't plateau enough
        validation_data=(X_test, Y_test),
        callbacks=[reduce_lr, checkpoints, early_stopping],
        sample_weight=sample_weights, verbose=0)
    
    #saving history
    metric_names = ["loss", "categorical_accuracy", "val_loss", "val_categorical_accuracy"]
    metric_df = pd.DataFrame() #creates empty dataframe
    for name in metric_names:
        metric_df[name] = history.history[name]
    metric_df.to_csv((report_path + test_name + "_history.csv"))

    #generating and saving best report by loading in best model and using it on the test set
    loaded_model = load_model((report_path + test_name + "_model.h5"))
    y_predictions = loaded_model.predict(X_test).argmax(axis=1)
    y_true = Y_test.argmax(axis=1)
    report = classification_report(y_true, y_predictions, output_dict=True, 
                                   zero_division=0, 
                                   target_names=label_classes)
    
    test_name = "NN_" + str(X_train.shape[0]) + "samples_" + str(X_train.shape[1]) + "Hz"
    #saving report
    with open((report_path + test_name + "_report.json"), 'w') as fp:
        json.dump(report, fp)
    
    return

#benchmark functions
#SVM function
def svm_func(X_train, X_test, Y_train, Y_train_onehot, Y_test, report_path):
    #generate sample weights
    #class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)
    #sample_weights = np.matmul(Y_train_onehot, class_weights)

    linear_svm = svm.SVC(kernel="linear", decision_function_shape="ovo")
    linear_svm.fit(X_train, Y_train)#, sample_weight=sample_weights)
    predictions = linear_svm.predict(X_test)
    report = classification_report(Y_test, predictions, output_dict=True, zero_division=0)
    test_name = "svm_" + str(X_train.shape[0]) + "samples_" + str(X_train.shape[1]) + "Hz"
    with open((report_path + test_name + "_report.json"), 'w') as fp:
        json.dump(report, fp)
    #saving model to pickled file
    with open((report_path + test_name + "_model.pkl"), "wb") as modelPickle:
        pickle.dump(linear_svm, modelPickle)
    return

#KNN function
#uses normal categorical text labels
def knn_func(X_train, X_test, Y_train, Y_test, report_path):
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, Y_train)
    predictions = knn_model.predict(X_test)
    report = classification_report(Y_test, predictions, output_dict=True, zero_division=0)
    test_name = "knn_" + str(X_train.shape[0]) + "samples_" + str(X_train.shape[1]) + "Hz"
    with open((report_path + test_name + "_report.json"), 'w') as fp:
        json.dump(report, fp)
    #saving model to pickled file
    with open((report_path + test_name + "_model.pkl"), "wb") as modelPickle:
        pickle.dump(knn_model, modelPickle)
    return 

#XGBoost Function
#uses binarized labels
def xgb_func(X_train, X_test, Y_train, Y_test, label_classes, report_path):
    y_integers = np.argmax(Y_train, axis=1)
    #class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_integers), y=y_integers)
    #sample_weights = np.matmul(Y_train, class_weights) 

    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, Y_train)#, sample_weight=sample_weights)
    predictions = xgb_clf.predict(X_test)
    report = classification_report(Y_test, predictions, output_dict=True, 
                                   zero_division=0, target_names=label_classes)
    test_name = "xgboost_" + str(X_train.shape[0]) + "samples_" + str(X_train.shape[1]) + "Hz"
    with open((report_path + test_name + "_report.json"), 'w') as fp:
        json.dump(report, fp)
    #saving model to pickled file
    with open((report_path + test_name + "_model.pkl"), "wb") as modelPickle:
        pickle.dump(xgb_clf, modelPickle)
    return
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
import random
import pandas as pd
import tensorflow as tf
import argparse
import keras_cv_attention_models
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense
from keras.regularizers import l2

parser = argparse.ArgumentParser(description = 'Running NN regression model')
parser.add_argument('--modell', dest="modell", default="ResNet50")
parser.add_argument('--image_size', dest='image_size', type=int, default=150)
parser.add_argument('--visit', dest="visit", default="0927")
parser.add_argument('--cv', dest="cv", default="cv1")
parser.add_argument('--cv_rate', dest="cv_rate", type=float, default=0.80)
parser.add_argument("--batch_size", dest="batch_size", type=int, default=100)
parser.add_argument("--epochs", dest="epochs", type=int, default=300)
parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.001)
parser.add_argument("--image_count_thr", dest="image_count_thr", type=int, default=100000)
parser.add_argument('--trainable', dest="trainable", default=True)
parser.add_argument("--seed", dest="seed", type=int, default=42)
parser.add_argument("--num_gpus", dest="num_gpus", type=int, default=2)
parser.add_argument("--opt", dest="opt", default="Adam")
parser.add_argument("--weight_decay", dest="weight_decay", type=float, default = 1e-4)
args = parser.parse_args()

#------------------------------------------------------------------------------------------------------------------
######################################################################
####################### Define hyperparameters #######################
######################################################################
modell = args.modell
image_size = args.image_size
visit = args.visit #"0718" "0801" "0815" "0912" "0927" "all"
cv = args.cv
cv_rate = args.cv_rate
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
patience = 20
trainable = args.trainable
image_count_thr = args.image_count_thr
seed = args.seed
num_gpus = args.num_gpus  # Number of GPUs you want to use
opt = args.opt
weight_decay = args.weight_decay

#------------------------------------------------------------------------------------------------------------------
######################################################################
######################### Print hyperparameter #######################
######################################################################
strategy = tf.distribute.MirroredStrategy()
print("######################################################################")
print("Model using: ", modell)
print("Seed is setting as:", seed)
print("image_size:", image_size)
print("visit:", visit)
print("cv:", cv)
print("batch_size:", batch_size)
print("epochs:", epochs)
print("learning_rate:", learning_rate)
print("patience:", patience)
print("Trainable:", trainable)
print("image_count_thr:", image_count_thr)
print(f"Optimizer{opt} with weight decay {weight_decay}")
print("GPU is using now:", tf.config.list_physical_devices('GPU'))

#------------------------------------------------------------------------------------------------------------------
######################################################################
########## Read in depth images and scale-based body weight ##########
######################################################################
os.chdir("UrFolder")
df1 = pd.read_csv("./labelled_depth_0718.csv", dtype={'Visit': str})
df2 = pd.read_csv("./labelled_depth_0801.csv", dtype={'Visit': str})
df3 = pd.read_csv("./labelled_depth_0815.csv", dtype={'Visit': str})
df4 = pd.read_csv("./labelled_depth_0829.csv", dtype={'Visit': str})
df5 = pd.read_csv("./labelled_depth_0912.csv", dtype={'Visit': str})
df6 = pd.read_csv("./labelled_depth_0927.csv", dtype={'Visit': str})

if cv == 'cv1':
    if visit == "0718":
        labelled_depth = df1
    elif visit == "0801":
        labelled_depth = df2
    elif visit == "0815":
        labelled_depth = df3
    elif visit == "0829":
        labelled_depth = df4
    elif visit == "0912":
        labelled_depth = df5
    elif visit == "0927":
        labelled_depth = df6
   
if cv == "cv2" or cv == 'cv2_0':
    if visit == "0801":
        labelled_depth = pd.concat([df1, df2], axis=0)
    elif visit == "0815":
        labelled_depth = pd.concat([df1, df2, df3], axis=0)
    elif visit == "0829":
        labelled_depth = pd.concat([df1, df2, df3, df4], axis=0)
    elif visit == "0912":
        labelled_depth = pd.concat([df1, df2, df3, df4, df5], axis=0)
    elif visit == "0927":
        labelled_depth = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)

#Remove outliers of scale based body weight.
weight_percentile = 2
weight_threshold = labelled_depth["Weights"].quantile(weight_percentile / 100)
labelled_depth = labelled_depth[labelled_depth["Weights"] >= weight_threshold]
print(f"Remove weight outliers by {weight_percentile}% quantile")

category_counts = labelled_depth['Bag_ID'].value_counts()
selected_rows = pd.DataFrame()
for category, count in category_counts.items():
    category_data = labelled_depth[labelled_depth['Bag_ID'] == category]
    if count <= image_count_thr:
        selected_rows = pd.concat([selected_rows, category_data])
    else:
        interval = count // image_count_thr
        selected_indices = np.arange(0, count, interval)[:image_count_thr]
        selected_rows = pd.concat([selected_rows, category_data.iloc[selected_indices]])
labelled_depth = selected_rows
print("Total images are ", labelled_depth.shape[0])

def read_images(labelled_depth, image_size):
    images = []
    img_paths = []
    for filename in labelled_depth["FilePath"]:
        if filename.endswith('.png'):  
            if os.path.exists(filename):
                img_path = filename
                img = tf.io.read_file(img_path)
                img = tf.image.decode_png(img, channels=3)  
                img = tf.image.resize_with_crop_or_pad(img, target_height=int(img.shape[1]), target_width=int(img.shape[1]))
                img = tf.image.resize(img, [image_size, image_size]) #Resize images
                images.append(img)
                img_paths.append(img_path)

    processed_images = tf.stack(images)
    processed_images = tf.cast(processed_images, tf.float32)
    processed_images /= 255.0
    print("DDDDDDDDDDDDDDDDDDDDDDDD")
    return processed_images, img_paths

#Cross-validation desgins.
if cv == "cv1":
    import random
    random.seed(seed)
    pig_n = np.unique(labelled_depth['Bag_ID']).shape[0]
    train_bag_id = random.sample(list(np.unique(labelled_depth["Bag_ID"])), int(pig_n*0.8))
    train_df = labelled_depth[labelled_depth["Bag_ID"].isin(train_bag_id)]
    test_df = labelled_depth[-labelled_depth["Bag_ID"].isin(train_bag_id)]

elif cv == 'cv2':
    cv_name = "CrossValidation" + str(cv_rate)
    train_df = labelled_depth[labelled_depth[cv_name] == "train"]
    test_df = labelled_depth[(labelled_depth[cv_name] == "test") & (labelled_depth["Visit"] == visit)]

elif cv == 'cv2_0':
    cv_name = "CrossValidation" + str(cv_rate)
    train_df = labelled_depth[
        (labelled_depth[cv_name] == "train") | 
        ((labelled_depth[cv_name] == "test") & (labelled_depth["Visit"] != visit))
        ]
    test_df = labelled_depth[(labelled_depth[cv_name] == "test") & (labelled_depth["Visit"] == visit)]

print("Now is reading training sets")
x_train, train_img_path = read_images(labelled_depth=train_df, image_size = image_size)
print("x_train shape is: ", x_train.shape, len(train_img_path))
y_train = train_df["Weights"].values
y_train = y_train.astype(np.float32)
print("y_train shape is: ", len(y_train))
print("Now is reading testing sets")
x_test, test_img_path = read_images(labelled_depth=test_df, image_size = image_size)
print("x_test shape is: ", x_test.shape, len(test_img_path))
y_test = test_df["Weights"].values
y_test = y_test.astype(np.float32)
print("y_test shape is: ", y_test.shape)

#------------------------------------------------------------------------------------------------------------------
######################################################################
###################### Train deep learning models ####################
######################################################################
with strategy.scope():

    def keras_cv_attention_model(modell):           
        if modell == "MobileViT_XXS":
            base_model = keras_cv_attention_models.mobilevit.MobileViT_XXS(pretrained="imagenet", num_classes=0, input_shape=(image_size, image_size, 3))
        elif modell == "MobileViT_S":
            base_model = keras_cv_attention_models.mobilevit.MobileViT_S(pretrained="imagenet", num_classes=0, input_shape=(image_size, image_size, 3))
        elif modell == "MobileViT_V2_050":
            base_model = keras_cv_attention_models.mobilevit.MobileViT_V2_050(pretrained="imagenet", num_classes=0, input_shape=(image_size, image_size, 3))
        elif modell == "MobileViT_V2_100":
            base_model = keras_cv_attention_models.mobilevit.MobileViT_V2_100(pretrained="imagenet", num_classes=0, input_shape=(image_size, image_size, 3))
        elif modell == "MobileViT_V2_150":
            base_model = keras_cv_attention_models.mobilevit.MobileViT_V2_150(pretrained="imagenet", num_classes=0, input_shape=(image_size, image_size, 3))
        elif modell == "MobileViT_V2_200":
            base_model = keras_cv_attention_models.mobilevit.MobileViT_V2_200(pretrained="imagenet", num_classes=0, input_shape=(image_size, image_size, 3))
        elif modell == "MobileNetV3Small050":
            base_model = keras_cv_attention_models.mobilenetv3_family.mobilenetv3.MobileNetV3Small050(pretrained="imagenet", num_classes=0, input_shape=(image_size, image_size, 3))
        elif modell == "MobileNetV3Large075":
            base_model = keras_cv_attention_models.mobilenetv3_family.mobilenetv3.MobileNetV3Large075(pretrained="imagenet", num_classes=0, input_shape=(image_size, image_size, 3))
        elif modell == "MobileNetV3Large100":
            base_model = keras_cv_attention_models.mobilenetv3_family.mobilenetv3.MobileNetV3Large100(pretrained="imagenet", num_classes=0, input_shape=(image_size, image_size, 3))
        else:
            raise ValueError(f"Model {modell} not recognized in keras_cv_attention_model function")

        return base_model

    
    def keras_model(modell):
        if modell == "ResNet50":
            base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
        elif modell == "MobileNet050":
            base_model = keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3), alpha=0.5)
        elif modell == "MobileNet100":
            base_model = keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3), alpha=1.0) 
        elif modell == "MobileNet075":
            base_model = keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3), alpha=0.75)   
        else:
            raise ValueError(f"Model {modell} not recognized in keras_model function")
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = trainable

        return base_model

   
    def load_model(model):
        try:
            return keras_cv_attention_model(model)
        except ValueError:
            return keras_model(model)

    base_model = load_model(model = modell) 
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dropout(0.5))  # Add dropout layer with 50% dropout rate
    model.add(Dense(1, activation='linear', kernel_regularizer=l2(0.001)))  # The output layer for regression with L2 regularization
   
    output_str = f"{modell}_visit_{visit}_image_size_{image_size}_trainable_{trainable}_batch_size_{batch_size}_epochs_{epochs}_lr_{learning_rate}_seed_{seed}_opt_{opt}_image_count_thr_{image_count_thr}"
    checkpoint_filepath = "/home/yebi/ComputerVision_PLF/Pig_BW/Run/tmp/checkpoint_" + output_str+'.h5'

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        patience=patience
        )
    
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=epochs//4,
    min_lr=1e-6
    )

    import pickle
    training_history = {'loss': [], 'mean_squared_error': [], 'val_loss': [], 'val_mean_squared_error': []}   

    if opt == "Adam":
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, epsilon=1e-05)
    if opt == "AdamW":
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, global_clipnorm=1.0)

    model.compile(optimizer=optimizer, 
                loss=keras.losses.MeanSquaredError(), 
                metrics=[keras.losses.MeanSquaredError()])


tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

history = model.fit(
    x=x_train,
    y=y_train,
    epochs=epochs, # The number of epochs
    batch_size=batch_size*num_gpus, # The batch size
    validation_split = 0.2,
    verbose=1,
    callbacks=[checkpoint_callback, reduce_lr_callback]
)

training_history['loss'].extend(history.history['loss'])
training_history['mean_squared_error'].extend(history.history['mean_squared_error'])
training_history['val_loss'].extend(history.history['val_loss'])
training_history['val_mean_squared_error'].extend(history.history['val_mean_squared_error'])

history_filename = "/home/yebi/ComputerVision_PLF/Pig_BW/Run/tmp/training_history_" + output_str + ".pkl"
with open(history_filename, 'wb') as file:
    pickle.dump(training_history, file)

#------------------------------------------------------------------------------------------------------------------
###################################################################### 
######### After model training, predict on testint set ###############
######################################################################
model.load_weights(checkpoint_filepath)

predicted_weights = np.squeeze(model.predict(x_test))
true_weights = y_test
prediction = pd.DataFrame({
    'predicted_Weights': predicted_weights,
    'true_Weights': true_weights
})
prediction.to_csv('/home/yebi/ComputerVision_PLF/Pig_BW/Run/tmp/prediction_'+output_str + '.csv', index=False)

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

rmse = mean_squared_error(true_weights, predicted_weights)**0.5
r2 = r2_score(true_weights,predicted_weights)
MAPE = mape(true_weights, predicted_weights)
MAE = mean_absolute_error(true_weights,predicted_weights)

import sys
original_stdout = sys.stdout
outputfile_path = f"/home/yebi/ComputerVision_PLF/Pig_BW/Run/tmp/output_" + output_str + ".txt"

print(outputfile_path)
try:
    with open(outputfile_path, 'w') as file:
        sys.stdout = file
        print('--------------------------')
        print("Model using: ", modell)
        print("Seed is setting as:", seed)
        print("image_size:", image_size)
        print("visit:", visit)
        print("cv:", cv)
        print("batch_size:", batch_size)
        print("epochs:", epochs)
        print("learning_rate:", learning_rate)
        print("patience:", patience)
        print("Trainable:", trainable)
        print("image_count_thr:", image_count_thr)
        if opt == "Adam":
            print(f"Optimizer: {opt}")
        if opt == "AdamW":
            print(f"Optimizer: {opt} with weight decay {weight_decay}")
        print("Total images are ", labelled_depth.shape[0])
        print('--------------------------')
        print('METRICS ON ENTIRE DATASET:')
        print('--------------------------')
        print("Test RMSE:\t{:.5f}".format(rmse))
        print("Test MAE:\t{:.5f}".format(MAE))
        print("Test R^2 Score:\t{:.5f}".format(r2))
        print("Test MAPE:\t{:.5f}%".format(MAPE))
        print('--------------------------')

        
finally:
    sys.stdout = original_stdout

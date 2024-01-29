import numpy as np
from PIL import Image


def index_to_word(index):
    all_labels = {0: 'Speed limit is 20km/h',
                  1: 'Speed limit is 30km/h',
                  2: 'Speed limit is 50km/h',
                  3: 'Speed limit is 60km/h',
                  4: 'Speed limit is 70km/h',
                  5: 'Speed limit is 80km/h',
                  6: 'End of speed limit is 80km/h',
                  7: 'Speed limit is 100km/h',
                  8: 'Speed limit is 120km/h',
                  9: 'No passing',
                  10: 'No passing veh is over 3.5 tons',
                  11: 'Right-of-way at intersection',
                  12: 'Priority road',
                  13: 'Yield',
                  14: 'Stop',
                  15: 'No vehicles',
                  16: 'Veh is bigger than 3.5 tons prohibited',
                  17: 'No entry',
                  18: 'General caution',
                  19: 'Dangerous curve left',
                  20: 'Dangerous curve right',
                  21: 'Double curve',
                  22: 'Bumpy road',
                  23: 'Slippery road',
                  24: 'Road narrows on the right',
                  25: 'Road work',
                  26: 'Traffic signals',
                  27: 'Pedestrians',
                  28: 'Children crossing',
                  29: 'Bicycles crossing',
                  30: 'Beware of ice/snow',
                  31: 'Wild animals crossing',
                  32: 'End speed + passing limits',
                  33: 'Turn right ahead',
                  34: 'Turn left ahead',
                  35: 'Ahead only',
                  36: 'Go straight or right',
                  37: 'Go straight or left',
                  38: 'Keep right',
                  39: 'Keep left',
                  40: 'Roundabout mandatory',
                  41: 'End of no passing',
                  42: 'End no passing veh is bigger than 3.5 tons'}
    return all_labels[index]


def to_pixel(img):
    img = Image.fromarray(np.uint8(img)).convert('RGB')
    img = img.resize((32, 32))
    img = np.asarray(img) / 255.  # normalize
    print("Img Shape:",img.shape)
    img = np.expand_dims(img, axis=0)  # add one more dimension to not facing dimension problem of models inputs shape
    return img


def predict(img, model):
    print("predicting...")
    pixel_value = to_pixel(img)  # get intensity values
    predicted_classid = model.predict(pixel_value)[0]  # predict label id probabilities by given model
    predicted_classid = np.argmax(predicted_classid)  # decide label id
    prediction_str = index_to_word(index=predicted_classid)  # get label_id's meaning
    return prediction_str

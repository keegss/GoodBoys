import sys
import io
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

from google.cloud import vision
from google.cloud.vision import types

def detect(img_path):

    print('Performing detection...')
    time.sleep(3)

    client = vision.ImageAnnotatorClient()

    file_name = os.path.join(os.path.dirname(__file__), img_path)

    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    objects = client.object_localization(image=image).localized_object_annotations

    count = 0
    for object_ in objects:
        if(object_.name == 'Dog'):
            count += 1

    print('Detection complete')
    print('-----')
    print(str(count) + " possible Good Boys detected")
    print('-----')
    time.sleep(3)

    return count

def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def goodBoyDetection(dog_count, img_path):
    print("Running Good Boy Detection Algorithm...")

    img = cv2.imread(img_path,1)

    # show original
    cv2.imshow('image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # now we go small
    lower_res = cv2.pyrDown(img)
    cv2.imshow("lower_res", lower_res)
    cv2.waitKey()
    even_lower_res = cv2.pyrDown(lower_res)
    cv2.imshow("lower_res", even_lower_res)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # show original
    cv2.imshow('image', img)
    cv2.waitKey(1000)

    # now we go big
    lower_res = cv2.pyrUp(img)
    cv2.imshow("lower_res", lower_res)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # atomize
    img2 = img.copy()
    for i in range(0, 10):
        res = np.random.shuffle(img2.reshape(-1, 3))
        cv2.imshow('image',img2)
        cv2.waitKey(1)

    cv2.imshow('image',img2)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    rainbow = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
    cv2.imshow('image', rainbow)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cool = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
    cv2.imshow('image', cool)
    cv2.waitKey()
    cv2.destroyAllWindows()

    rainbow = cv2.applyColorMap(img, cv2.COLORMAP_PINK)
    cv2.imshow('image', rainbow)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    total = 20
    i = 0
    while i <= total:
        progress(i, total, status='Aggregating results')
        time.sleep(0.5)
        i += 1

    print('\n-----')
    print('RESULTS')
    print('-----')
    for i in range(1, dog_count+1):
        print('DOG %d: Good Boy' % i)
    print("Algorithm Complete")

def main():
    # google credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gudbois-b4a1ba2957a8.json'

    img_path = sys.argv[1]

    image = cv2.imread(img_path)
    cv2.imshow('Detection Area', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    dog_count = detect(img_path)
    goodBoyDetection(dog_count, img_path)

if __name__ == "__main__":
    main()

import cv2
import numpy as np
# import matplotlib.pyplot as plt


def cartoon_image(img):

    # 1) Edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # 2) Color
    color = cv2.bilateralFilter(img, 9, 300, 300)

    # 3) Cartoon
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon


def quantize(image, number_of_colors):

    NCLUSTERS = number_of_colors
    NROUNDS = 1

    height, width, channels = image.shape
    samples = np.zeros([height*width, 3], dtype=np.float32)
    count = 0

    for x in range(height):
        for y in range(width):
            samples[count] = image[x][y]  # BGR color
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
                                              NCLUSTERS,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS +
                                               cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                              NROUNDS,
                                              cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    image2 = res.reshape((image.shape))

    return image2


def sat_val_increase(image, satuartion_plus, value_plus):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            if image[i][j][2] > 10:
                if image[i][j][2] + value_plus < 255:
                    image[i][j][2] += value_plus
                else:
                    image[i][j][2] = 255

            if image[i][j][1] > 10:
                if image[i][j][1] + satuartion_plus < 255:
                    image[i][j][1] += satuartion_plus
                else:
                    image[i][j][1] = 255

    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def scale_hard(image, scale_factor):

    if scale_factor == 1:
        return image

    return cv2.resize(
        image, (image.shape[1] * scale_factor, image.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)


input_file = "input.jpg"

orig = cv2.imread(input_file)

# orig = cv2.resize(orig, (400, 400))

number_of_colors = 16

satuartion_plus = 70
value_plus = 40

scale_factor = 1

quant = quantize(orig, number_of_colors)
cartoon = cartoon_image(quant)
sat_val = sat_val_increase(cartoon, satuartion_plus, value_plus)
last_quant = quantize(sat_val, number_of_colors)
last = scale_hard(last_quant, scale_factor)

# cv2.imshow("cartoon.png", last_quant)
cv2.imwrite("cartoon.png", last)

cv2.waitKey(0)

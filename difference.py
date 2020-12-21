import cv2 as cv
import numpy as np


def minmax(array):
    array = array - np.min(array) if np.min(array) < 0 else array
    array = array / np.max(array) if np.max(array) > 1 else array
    return array


def get_difference(path, matrix):
    kernel_h, kernel_w = np.shape(matrix)
    img = cv.imread(path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (3, 3), 1)
    img_h, img_w = np.shape(blur_img)
    cv.imshow('blur', blur_img)
    cv.waitKey(0)
    diff_img = []
    for hf in range(kernel_h):
        for wf in range(kernel_w):
            diff = blur_img[hf:img_h - kernel_h + hf + 1:1, wf:img_w - kernel_w + wf + 1:1]
            diff_img.append(diff)
            cv.imshow('kernel', diff)
            cv.waitKey(0)
    matrix = np.reshape(matrix, (-1, 1, 1))
    print(matrix, np.shape(diff_img))
    diff_img = np.multiply(diff_img, matrix)
    diff_img = np.sum(diff_img, 0)
    cv.imshow('diff', minmax(diff_img))
    cv.waitKey(0)
    return diff_img


if __name__ == '__main__':
    matrix = np.asarray([
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0]
    ])
    dx = get_difference('C:/task.png', matrix)
    cv.imwrite('D:/dx.png', np.round(minmax(dx) * 255))
    matrix = np.asarray([
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0]
    ])
    dy = get_difference('C:/task.png', matrix)
    cv.imwrite('D:/dy.png', np.round(minmax(dy) * 255))

    d = np.sqrt(np.square(dx) + np.square(dy))
    cv.imwrite('D:/d.png', np.round(minmax(d) * 255))
    cv.imshow('d', minmax(d))
    cv.waitKey(0)

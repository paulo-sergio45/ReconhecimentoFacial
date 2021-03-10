import cv2
import numpy as np


def alinhaFace(img, posi):
    er = posi["right_eye"]
    el = posi["left_eye"]
    delta_x = er[0] - el[0]
    delta_y = er[1] - el[1]
    angle = np.arctan(delta_y / delta_x)
    angle = (angle * 180) / np.pi

    h, w = img.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, (angle), 1.0)

    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def normalizaFace(imagen):
    alow = imagen.min()
    ahigh = imagen.max()
    amax = 255
    amin = 0

    # calcula alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha

    new_image = cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

    return new_image

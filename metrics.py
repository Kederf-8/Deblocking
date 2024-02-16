import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def calculate_mse(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))

    if img1 is None or img2 is None:
        print("Impossibile leggere una delle immagini.")
        return None

    if img1.shape != img2.shape:
        raise ValueError("Le dimensioni delle immagini non corrispondono.")

    # Calcola il Mean Squared Error (MSE)
    mse = np.sum((img1 - img2) ** 2) / float(img1.shape[0] * img1.shape[1])

    return mse


def calculate_psnr(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Impossibile leggere una delle immagini.")
        return None

    if img1.shape != img2.shape:
        print("Le dimensioni delle immagini non corrispondono.")
        return None

    # Calcola il PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


def calculate_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Impossibile leggere una delle immagini.")
        return None

    if img1.shape != img2.shape:
        print("Le dimensioni delle immagini non corrispondono.")
        return None

    # Converti le immagini in scala di grigi
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calcola l'indice SSIM
    indice_ssim, _ = ssim(img1_gray, img2_gray, full=True)

    return indice_ssim

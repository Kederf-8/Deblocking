import cv2
import pywt
import numpy as np


def wavelet_filter(image_path, output_path):
    # Carica l'immagine JPEG a colori
    img = cv2.imread(image_path)

    # Split dei canali RGB
    b, g, r = cv2.split(img)

    # Applica la trasformata wavelet di Haar a ciascun canale
    coeffs_b = pywt.dwt2(b, "haar")
    coeffs_g = pywt.dwt2(g, "haar")
    coeffs_r = pywt.dwt2(r, "haar")

    # Modifica i coefficienti a tua discrezione, ad esempio, quantizzando in modo più aggressivo
    coeffs_b_modified = [np.round(c) for c in coeffs_b]
    coeffs_g_modified = [np.round(c) for c in coeffs_g]
    coeffs_r_modified = [np.round(c) for c in coeffs_r]

    # Ricostruisci i canali dalla trasformata wavelet modificata
    b_wavelet = pywt.idwt2(coeffs_b_modified, "haar")
    g_wavelet = pywt.idwt2(coeffs_g_modified, "haar")
    r_wavelet = pywt.idwt2(coeffs_r_modified, "haar")

    # Unisci i canali per ottenere l'immagine finale
    img_wavelet = cv2.merge(
        (
            b_wavelet.astype(np.uint8),
            g_wavelet.astype(np.uint8),
            r_wavelet.astype(np.uint8),
        )
    )

    # Assicurati che l'immagine risultante abbia le stesse dimensioni di quella di input
    img_wavelet = img_wavelet[: img.shape[0], : img.shape[1], :]

    # Salva l'immagine risultante
    cv2.imwrite(output_path, img_wavelet)

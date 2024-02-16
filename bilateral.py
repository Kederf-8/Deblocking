import cv2


def bilateral_filter(image_path, output_path, d=5, sigma_color=15, sigma_space=15):
    # Leggi l'immagine
    img = cv2.imread(image_path)

    # Applica il filtro bilaterale
    filtered_img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    # Salva l'immagine filtrata
    cv2.imwrite(output_path, filtered_img)

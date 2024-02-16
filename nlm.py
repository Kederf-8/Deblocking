import cv2


def nlm_filter(image_path, save_path, h_value=10, sigma_color=1, sigma_space=1):
    # Leggi l'immagine
    img = cv2.imread(image_path)

    # Applica il filtro Non-Local Means (NLM) separatamente su ciascun canale di colore
    b, g, r = cv2.split(img)

    b_denoised = cv2.fastNlMeansDenoising(b, None, h=h_value)
    g_denoised = cv2.fastNlMeansDenoising(g, None, h=h_value)
    r_denoised = cv2.fastNlMeansDenoising(r, None, h=h_value)

    # Combina i canali denoised
    denoised_img = cv2.merge([b_denoised, g_denoised, r_denoised])

    # Applica l'operazione di smoothing per rimuovere la blocchettizzazione
    smoothed_img = cv2.GaussianBlur(denoised_img, (5, 5), sigma_color, sigma_space)

    # Salva l'immagine risultante
    cv2.imwrite(save_path, smoothed_img)

def nlm_filter2(image_path, save_path, h_value=10):
    # Leggi l'immagine
    img = cv2.imread(image_path)

    # Applica il filtro Non-Local Means (NLM) all'intera immagine
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, h=h_value)

    # Salva l'immagine risultante
    cv2.imwrite(save_path, denoised_img)


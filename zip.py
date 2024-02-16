from PIL import Image


def compress_jpeg(input_image_path, output_image_path, quality):
    # Apri l'immagine
    original_image = Image.open(input_image_path)

    # Salva l'immagine compressa in formato JPEG
    original_image.save(output_image_path, "JPEG", quality=quality)

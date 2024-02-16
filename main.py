from tabulate import tabulate

from bilateral import bilateral_filter
from metrics import calculate_mse, calculate_psnr, calculate_ssim
from nlm import nlm_filter
from wavelet import wavelet_filter
from zip import compress_jpeg


def process_and_compare(original_path, image, compression_ratio, output_folder):
    compressed_path = f"{output_folder}/{image}_zip_{compression_ratio}.jpg"
    bilateral_path = f"bilateral_images/{image}_zip_{compression_ratio}_bilateral.jpg"
    nlm_path = f"nlm_images/{image}_zip_{compression_ratio}_nlm.jpg"
    wavelet_path = f"wavelet_images/{image}_zip_{compression_ratio}_wavelet.jpg"

    # Compressione JPEG
    compress_jpeg(original_path, compressed_path, compression_ratio)

    # Filtri
    bilateral_filter(compressed_path, bilateral_path)
    nlm_filter(compressed_path, nlm_path)
    wavelet_filter(compressed_path, wavelet_path)

    # Calcola l'MSE per ogni filtro
    mse_zipped = calculate_mse(original_path, compressed_path)
    mse_bilateral = calculate_mse(original_path, bilateral_path)
    mse_nlm = calculate_mse(original_path, nlm_path)
    mse_wavelet = calculate_mse(original_path, wavelet_path)

    # Calcola l'SSIM per ogni filtro
    ssim_zipped = calculate_ssim(original_path, compressed_path)
    ssim_bilateral = calculate_ssim(original_path, bilateral_path)
    ssim_nlm = calculate_ssim(original_path, nlm_path)
    ssim_wavelet = calculate_ssim(original_path, wavelet_path)

    # Calcola l'PSNR per ogni filtro
    psnr_zipped = calculate_psnr(original_path, compressed_path)
    psnr_bilateral = calculate_psnr(original_path, bilateral_path)
    psnr_nlm = calculate_psnr(original_path, nlm_path)
    psnr_wavelet = calculate_psnr(original_path, wavelet_path)

    table = tabulate(
        [
            ["Metrica", "Senza Filtro", "Bilaterale", "NLM", "Wavelet"],
            [
                "MSE",
                round(mse_zipped, 2),
                round(mse_bilateral, 2),
                round(mse_nlm, 2),
                round(mse_wavelet, 2),
            ],
            [
                "SSIM",
                round(ssim_zipped, 3),
                round(ssim_bilateral, 3),
                round(ssim_nlm, 3),
                round(ssim_wavelet, 3),
            ],
            [
                "PSNR",
                round(psnr_zipped, 2),
                round(psnr_bilateral, 2),
                round(psnr_nlm, 2),
                round(psnr_wavelet, 2),
            ],
        ],
        tablefmt="pretty",
    )

    print(
        f"\nAnalisi per l'immagine {image}, compressione con {compression_ratio}% di qualità:\n"
    )
    print(table)


output_folder = "blocked_images"
compression_ratios = [10, 20, 30]
images = ["mountain", "bedroom", "jobs", "rose", "living", "colors", "everest"]

for image in images:
    input_path = f"input_images/{image}.jpg"
    print(f"Immagine {input_path}")
    for ratio in compression_ratios:
        process_and_compare(input_path, image, ratio, output_folder)
    print("-----------------------------------------------------")

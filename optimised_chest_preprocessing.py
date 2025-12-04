"""
PACE 2.0 - FINAL OPTIMIZED for Chest X-Rays
Preserves rib bone visibility while enhancing lung details

KEY FINDING: β=0 (exclude homomorphic residual) is critical!
- Prevents over-normalization of bright areas (ribs)
- Maintains bone visibility
- Still enhances dark lung fields

Optimal parameters:
- gamma = 1.8 (brightness)
- gamma_h = 2.0 (homomorphic filtering)
- beta = 0 (NO residual - this preserves ribs!) ← KEY!
- clip_limit = 0.02 (CLAHE)
"""

import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class PACE2Pipeline:
    """PACE 2.0 - Optimized for Chest X-Rays with Rib Preservation"""

    def __init__(self, n_bimfs=10, verbose=False):
        self.n_bimfs = n_bimfs
        self.verbose = verbose

    def _log(self, message):
        if self.verbose:
            print(f"[PACE2.0] {message}")

    def fabemd(self, image, n_imfs=10):
        """Fast and Adaptive Bi-dimensional Empirical Mode Decomposition"""
        self._log("Applying FABEMD decomposition...")
        bimfs = []
        residual = image.copy()

        for i in range(n_imfs):
            kernel_size = 3 + (i * 2)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            upper_env = cv2.dilate(residual, kernel)
            lower_env = cv2.erode(residual, kernel)
            mean_env = (upper_env + lower_env) / 2.0
            bimf = residual - mean_env
            bimfs.append(bimf)
            residual = mean_env
            if np.std(residual) < 0.01:
                break

        return bimfs, residual

    def homomorphic_filter(self, image, gamma_h=1.5, gamma_l=0.5, c=1.0, d0=10):
        """Apply Homomorphic Filtering"""
        self._log("Applying Homomorphic Filter...")
        image_safe = np.maximum(image, 1e-6)
        log_image = np.log1p(image_safe)
        f_transform = np.fft.fft2(log_image)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        x = np.arange(cols) - ccol
        y = np.arange(rows) - crow
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X ** 2 + Y ** 2)
        H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D ** 2 / (d0 ** 2)))) + gamma_l
        f_filtered = f_shift * H
        f_ishift = np.fft.ifftshift(f_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        image_filtered = np.real(image_filtered)
        result = np.expm1(image_filtered)
        result = np.clip(result, 0, 1)
        return result

    def calculate_bimf_energy(self, bimfs):
        """Calculate energy of each BIMF"""
        self._log("Calculating BIMF energies...")
        energies = []
        for bimf in bimfs:
            energy = np.sum(bimf ** 2)
            energies.append(energy)
        noise_threshold = min(3, len(bimfs) // 3)
        return energies, noise_threshold

    def non_local_means_denoise(self, image, h=3, template_window_size=7, search_window_size=21):
        """Apply Non-Local Means denoising"""
        image_uint8 = (image * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(
            image_uint8, h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
        return denoised.astype(np.float32) / 255.0

    def reconstruct_image(self, bimfs, residual, beta, gamma_h, noise_threshold):
        """Reconstruct image from BIMFs"""
        self._log(f"Reconstructing image (β={beta} - {'includes' if beta else 'EXCLUDES'} residual)...")
        denoised_bimfs = []
        for i, bimf in enumerate(bimfs):
            if i < noise_threshold:
                denoised = self.non_local_means_denoise(
                    (bimf - bimf.min()) / (bimf.max() - bimf.min() + 1e-8)
                )
                denoised = denoised * (bimf.max() - bimf.min()) + bimf.min()
                denoised_bimfs.append(denoised)
            else:
                denoised_bimfs.append(bimf)

        IE = np.sum(denoised_bimfs, axis=0)

        if beta > 0:
            residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
            IHMF = self.homomorphic_filter(residual_norm, gamma_h=gamma_h)
            IL = IE + beta * IHMF
        else:
            # β=0: Skip homomorphic residual to preserve bright areas (ribs)
            IL = IE

        IL = (IL - IL.min()) / (IL.max() - IL.min() + 1e-8)
        return IL

    def gamma_correction(self, image, gamma=1.0):
        """Apply gamma correction"""
        return np.power(image, gamma)

    def clahe(self, image, clip_limit=0.01, tile_grid_size=(4, 4)):
        """Apply CLAHE"""
        image_uint8 = (image * 255).astype(np.uint8)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit * 100, tileGridSize=tile_grid_size)
        enhanced = clahe_obj.apply(image_uint8)
        return enhanced.astype(np.float32) / 255.0

    def calculate_entropy(self, image):
        """Calculate entropy"""
        image_uint8 = (image * 255).astype(np.uint8)
        hist, _ = np.histogram(image_uint8, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def calculate_cii(self, original, processed):
        """Calculate CII"""
        c_original = np.std(original)
        c_processed = np.std(processed)
        if c_original == 0:
            return 1.0
        return c_processed / c_original

    def calculate_eme(self, image, k1=4, k2=4):
        """Calculate EME"""
        h, w = image.shape
        block_h = h // k1
        block_w = w // k2
        eme = 0.0
        count = 0

        for i in range(k1):
            for j in range(k2):
                block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                I_max = block.max()
                I_min = block.min()
                if I_min > 0:
                    contrast_ratio = I_max / (I_min + 1e-6)
                    eme += 20 * np.log(contrast_ratio + 1e-6)
                    count += 1

        if count == 0:
            return 0.0
        return eme / count

    def process_final(self, image):
        """
        FINAL OPTIMIZED MODE - 2-3 seconds

        ✅ Preserves rib bone visibility
        ✅ Enhances lung details
        ✅ Best overall quality

        Key parameter: β=0 (excludes homomorphic residual)
        This prevents over-normalization of bright areas (ribs)
        """
        self._log("=" * 60)
        self._log("PACE 2.0 - FINAL OPTIMIZED (Rib-Preserving)")
        self._log("=" * 60)

        # Normalize
        if image.dtype == np.uint8:
            image_norm = image.astype(np.float32) / 255.0
        else:
            image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Decomposition
        bimfs, residual = self.fabemd(image_norm, n_imfs=self.n_bimfs)
        energies, noise_threshold = self.calculate_bimf_energy(bimfs)

        # FINAL OPTIMAL parameters
        gamma = 1.8          # Good brightness
        gamma_h = 2.0        # Strong homomorphic filtering
        beta = 0             # ← KEY! Exclude residual to preserve ribs
        clip_limit = 0.02    # Strong CLAHE

        self._log(f"Key setting: β={beta} (EXCLUDES homomorphic residual to preserve ribs)")

        # Process
        IL = self.reconstruct_image(bimfs, residual, beta, gamma_h, noise_threshold)
        I_gamma = self.gamma_correction(IL, gamma)
        enhanced = self.clahe(I_gamma, clip_limit=clip_limit)

        # Metrics
        ent = self.calculate_entropy(enhanced)
        cii = self.calculate_cii(image_norm, enhanced)
        eme = self.calculate_eme(enhanced)

        params = {
            'gamma': gamma, 'gamma_h': gamma_h, 'beta': beta, 'clip_limit': clip_limit,
            'ent': ent, 'cii': cii, 'eme': eme,
            'score': 0.3 * ent / 8.0 + 0.3 * cii / 2.0 + 0.4 * eme / 10.0
        }

        self._log(f"Metrics: ENT={ent:.4f}, CII={cii:.4f}, EME={eme:.4f}")
        self._log("=" * 60)
        self._log("Processing Complete - Ribs Preserved!")
        self._log("=" * 60)

        return enhanced, params


# ==================== SIMPLE USAGE ====================

if __name__ == "__main__":


    import os

    parent_directory = "/data2/rohit_gene_prediction/chest_testing/CheXpert-v1.0-small/train/"

    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        for study in os.listdir(item_path):
            file_folder_path = os.path.join(item_path, study)
            for root, dirs, files in os.walk(file_folder_path):
                for file_name in files:
                    input_file = os.path.join(root, file_name)
                    print(f"Processing file: {input_file}")
                    output_file = os.path.join(root, "processed" + file_name)

                    # # Input and output files
                    # input_file = "view1_frontal.jpg"  # CHANGE THIS
                    # output_file = "view1_frontal_enhanced.jpg"  # CHANGE THIS

                    print("="*70)
                    print("PACE 2.0 - FINAL OPTIMIZED (Preserves Ribs + Enhances Lungs)")
                    print("="*70)

                    # Load image
                    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

                    if image is None:
                        print(f"ERROR: Could not load '{input_file}'")
                        print("\nPlease edit the script and change:")
                        print("  input_file = 'your_xray.jpg'")
                        print("  output_file = 'your_xray_enhanced.jpg'")
                    else:
                        if ("processed" + file_name) not in files:

                            print(f"✓ Loaded: {input_file} ({image.shape[1]}x{image.shape[0]})")

                            # Process
                            print("\nProcessing (2-3 sec)...")
                            pipeline = PACE2Pipeline(verbose=True)
                            enhanced, params = pipeline.process_final(image)

                            # Save
                            enhanced_uint8 = (enhanced * 255).astype('uint8')
                            success = cv2.imwrite(output_file, enhanced_uint8)
                        else:
                            print(f"{file_name} was processed and present")


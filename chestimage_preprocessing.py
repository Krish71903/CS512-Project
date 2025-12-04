"""
PACE 2.0: Pipeline for Advanced Contrast Enhancement
Implementation based on the paper:
"Effective processing pipeline PACE 2.0 for enhancing chest x-ray contrast and diagnostic interpretability"
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings('ignore')


class PACE2Pipeline:
    """
    PACE 2.0 - Pipeline for Advanced Contrast Enhancement for Chest X-Ray images

    This implementation combines:
    1. FABEMD - Fast and Adaptive Bi-dimensional Empirical Mode Decomposition
    2. HMF - Homomorphic Filtering
    3. BIMF Energy Calculation
    4. Non-Local Means Denoising
    5. Gamma Correction
    6. CLAHE - Contrast Limited Adaptive Histogram Equalization
    7. Multi-Objective Optimization
    """

    def __init__(self, n_bimfs=10, verbose=False):
        """
        Initialize PACE 2.0 pipeline

        Args:
            n_bimfs: Number of BIMFs to extract (default: 10)
            verbose: Print progress information
        """
        self.n_bimfs = n_bimfs
        self.verbose = verbose

        # MOO parameter ranges
        self.gamma_range = np.arange(0.5, 2.1, 0.1)  # 16 values
        self.gh_range = np.arange(1.0, 2.6, 0.5)  # 4 values
        self.beta_range = [0, 1]  # 2 values
        self.clahe_clip_range = [0.01, 0.02]  # 2 values

    def _log(self, message):
        """Print message if verbose is enabled"""
        if self.verbose:
            print(f"[PACE2.0] {message}")

    def fabemd(self, image, n_imfs=10):
        """
        Fast and Adaptive Bi-dimensional Empirical Mode Decomposition
        Decomposes image into BIMFs and residual

        Args:
            image: Input grayscale image (normalized 0-1)
            n_imfs: Number of IMFs to extract

        Returns:
            bimfs: List of BIMFs
            residual: Residual image
        """
        self._log("Applying FABEMD decomposition...")

        bimfs = []
        residual = image.copy()

        for i in range(n_imfs):
            # Simplified BEMD using morphological operations
            # Find local maxima and minima
            kernel_size = 3 + (i * 2)  # Increasing kernel size
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Envelope estimation using dilation and erosion
            upper_env = cv2.dilate(residual, kernel)
            lower_env = cv2.erode(residual, kernel)

            # Mean envelope
            mean_env = (upper_env + lower_env) / 2.0

            # Extract BIMF
            bimf = residual - mean_env
            bimfs.append(bimf)

            # Update residual
            residual = mean_env

            # Stop if residual is monotonic (approximation)
            if np.std(residual) < 0.01:
                break

        return bimfs, residual

    def homomorphic_filter(self, image, gamma_h=1.5, gamma_l=0.5, c=1.0, d0=10):
        """
        Apply Homomorphic Filtering for illumination normalization

        Args:
            image: Input image
            gamma_h: High frequency gain
            gamma_l: Low frequency gain
            c: Sharpness constant
            d0: Cutoff frequency

        Returns:
            Filtered image
        """
        self._log("Applying Homomorphic Filter...")

        # Avoid log(0)
        image_safe = np.maximum(image, 1e-6)

        # Take logarithm
        log_image = np.log1p(image_safe)

        # Apply FFT
        f_transform = np.fft.fft2(log_image)
        f_shift = np.fft.fftshift(f_transform)

        # Create High-Frequency Emphasis Filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Create meshgrid for distance calculation
        x = np.arange(cols) - ccol
        y = np.arange(rows) - crow
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X ** 2 + Y ** 2)

        # HEF Filter
        H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D ** 2 / (d0 ** 2)))) + gamma_l

        # Apply filter
        f_filtered = f_shift * H

        # Inverse FFT
        f_ishift = np.fft.ifftshift(f_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        image_filtered = np.real(image_filtered)

        # Exponentiate
        result = np.expm1(image_filtered)

        # Normalize
        result = np.clip(result, 0, 1)

        return result

    def calculate_bimf_energy(self, bimfs):
        """
        Calculate energy of each BIMF to determine significance

        Args:
            bimfs: List of BIMFs

        Returns:
            energies: Energy of each BIMF
            noise_threshold: Index of first significant BIMF
        """
        self._log("Calculating BIMF energies...")

        energies = []
        for bimf in bimfs:
            energy = np.sum(bimf ** 2)
            energies.append(energy)

        # Simple heuristic: first 3 BIMFs typically contain noise
        # This could be refined with more sophisticated analysis
        noise_threshold = min(3, len(bimfs) // 3)

        return energies, noise_threshold

    def non_local_means_denoise(self, image, h=3, template_window_size=7, search_window_size=21):
        """
        Apply Non-Local Means denoising

        Args:
            image: Input image (0-1 range)
            h: Filter strength
            template_window_size: Size of template patch
            search_window_size: Size of search area

        Returns:
            Denoised image
        """
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)

        # Apply NLM denoising
        denoised = cv2.fastNlMeansDenoising(
            image_uint8,
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )

        # Convert back to float
        return denoised.astype(np.float32) / 255.0

    def reconstruct_image(self, bimfs, residual, beta, gamma_h, noise_threshold):
        """
        Reconstruct image from BIMFs and residual

        Args:
            bimfs: List of BIMFs
            residual: Residual image
            beta: Weight for filtered residual (0 or 1)
            gamma_h: Parameter for homomorphic filter
            noise_threshold: Number of BIMFs to denoise

        Returns:
            Reconstructed image
        """
        self._log(f"Reconstructing image (beta={beta}, noise_threshold={noise_threshold})...")

        # Denoise first R BIMFs
        denoised_bimfs = []
        for i, bimf in enumerate(bimfs):
            if i < noise_threshold:
                # Denoise noise-heavy BIMFs
                denoised = self.non_local_means_denoise(
                    (bimf - bimf.min()) / (bimf.max() - bimf.min() + 1e-8)
                )
                # Rescale back
                denoised = denoised * (bimf.max() - bimf.min()) + bimf.min()
                denoised_bimfs.append(denoised)
            else:
                denoised_bimfs.append(bimf)

        # Sum all BIMFs
        IE = np.sum(denoised_bimfs, axis=0)

        # Apply homomorphic filter to residual
        residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
        IHMF = self.homomorphic_filter(residual_norm, gamma_h=gamma_h)

        # Reconstruct
        IL = IE + beta * IHMF

        # Normalize to [0, 1]
        IL = (IL - IL.min()) / (IL.max() - IL.min() + 1e-8)

        return IL

    def gamma_correction(self, image, gamma=1.0):
        """
        Apply gamma correction

        Args:
            image: Input image (0-1 range)
            gamma: Gamma value

        Returns:
            Gamma corrected image
        """
        return np.power(image, gamma)

    def clahe(self, image, clip_limit=0.01, tile_grid_size=(4, 4)):
        """
        Apply Contrast Limited Adaptive Histogram Equalization

        Args:
            image: Input image (0-1 range)
            clip_limit: Clipping limit
            tile_grid_size: Size of grid for histogram equalization

        Returns:
            CLAHE enhanced image
        """
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)

        # Create CLAHE object
        clahe_obj = cv2.createCLAHE(
            clipLimit=clip_limit * 100,  # OpenCV uses different scale
            tileGridSize=tile_grid_size
        )

        # Apply CLAHE
        enhanced = clahe_obj.apply(image_uint8)

        # Convert back to float
        return enhanced.astype(np.float32) / 255.0

    def calculate_entropy(self, image):
        """Calculate image entropy (ENT metric)"""
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)

        # Calculate histogram
        hist, _ = np.histogram(image_uint8, bins=256, range=(0, 256))
        hist = hist / hist.sum()

        # Remove zeros
        hist = hist[hist > 0]

        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))

        return entropy

    def calculate_cii(self, original, processed):
        """
        Calculate Contrast Improvement Index

        Args:
            original: Original image
            processed: Processed image

        Returns:
            CII value
        """

        def calculate_contrast(img):
            # Simple contrast measure using standard deviation
            return np.std(img)

        c_original = calculate_contrast(original)
        c_processed = calculate_contrast(processed)

        if c_original == 0:
            return 1.0

        return c_processed / c_original

    def calculate_eme(self, image, k1=4, k2=4):
        """
        Calculate Effective Measure of Enhancement (EME)

        Args:
            image: Input image
            k1, k2: Number of blocks in each dimension

        Returns:
            EME value
        """
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

    def multi_objective_optimization(self, original, bimfs, residual, noise_threshold):
        """
        Multi-Objective Optimization to find best parameters

        Args:
            original: Original image
            bimfs: List of BIMFs
            residual: Residual image
            noise_threshold: Noise threshold index

        Returns:
            best_image: Best enhanced image
            best_params: Best parameters
            best_score: Best score
        """
        self._log("Running Multi-Objective Optimization...")

        best_score = -np.inf
        best_image = None
        best_params = None

        total_combinations = (len(self.gamma_range) * len(self.gh_range) *
                              len(self.beta_range) * len(self.clahe_clip_range))

        combination_count = 0

        for gamma in self.gamma_range:
            for gamma_h in self.gh_range:
                for beta in self.beta_range:
                    for clip_limit in self.clahe_clip_range:
                        combination_count += 1

                        if self.verbose and combination_count % 20 == 0:
                            print(f"  Testing combination {combination_count}/{total_combinations}")

                        # Reconstruct image
                        IL = self.reconstruct_image(bimfs, residual, beta, gamma_h, noise_threshold)

                        # Apply gamma correction
                        I_gamma = self.gamma_correction(IL, gamma)

                        # Apply CLAHE
                        I_clahe = self.clahe(I_gamma, clip_limit=clip_limit)

                        # Calculate metrics
                        ent = self.calculate_entropy(I_clahe)
                        cii = self.calculate_cii(original, I_clahe)
                        eme = self.calculate_eme(I_clahe)

                        # Combined score (weighted sum)
                        # Normalize and combine
                        score = 0.3 * ent / 8.0 + 0.3 * cii / 2.0 + 0.4 * eme / 10.0

                        if score > best_score:
                            best_score = score
                            best_image = I_clahe
                            best_params = {
                                'gamma': gamma,
                                'gamma_h': gamma_h,
                                'beta': beta,
                                'clip_limit': clip_limit,
                                'ent': ent,
                                'cii': cii,
                                'eme': eme,
                                'score': score
                            }

        self._log(f"Best score: {best_score:.4f}")
        self._log(f"Best params: gamma={best_params['gamma']:.2f}, "
                  f"gamma_h={best_params['gamma_h']:.2f}, "
                  f"beta={best_params['beta']}, "
                  f"clip_limit={best_params['clip_limit']:.3f}")
        self._log(f"Metrics: ENT={best_params['ent']:.4f}, "
                  f"CII={best_params['cii']:.4f}, "
                  f"EME={best_params['eme']:.4f}")

        return best_image, best_params, best_score

    def process(self, image):
        """
        Process an image through the complete PACE 2.0 pipeline

        Args:
            image: Input grayscale image (uint8 or float, any range)

        Returns:
            enhanced: Enhanced image (float32, 0-1 range)
            params: Parameters used for enhancement
        """
        self._log("=" * 60)
        self._log("Starting PACE 2.0 Pipeline")
        self._log("=" * 60)

        # Normalize input to [0, 1]
        if image.dtype == np.uint8:
            image_norm = image.astype(np.float32) / 255.0
        else:
            image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Step 1: FABEMD
        bimfs, residual = self.fabemd(image_norm, n_imfs=self.n_bimfs)

        # Step 2: Calculate BIMF energies
        energies, noise_threshold = self.calculate_bimf_energy(bimfs)

        # Step 3-7: Multi-Objective Optimization
        enhanced, params, score = self.multi_objective_optimization(
            image_norm, bimfs, residual, noise_threshold
        )

        self._log("=" * 60)
        self._log("PACE 2.0 Pipeline Complete")
        self._log("=" * 60)

        return enhanced, params


def process_image_file(input_path, output_path, verbose=True):
    """
    Process a single image file

    Args:
        input_path: Path to input image
        output_path: Path to save enhanced image
        verbose: Print progress
    """
    # Load image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image: {input_path}")

    # Create pipeline
    pipeline = PACE2Pipeline(verbose=verbose)

    # Process
    enhanced, params = pipeline.process(image)

    # Convert to uint8
    enhanced_uint8 = (enhanced * 255).astype(np.uint8)

    # Save
    cv2.imwrite(output_path, enhanced_uint8)

    print(f"\nSaved enhanced image to: {output_path}")

    return enhanced_uint8, params


if __name__ == "__main__":

    import os

    parent_directory = "/data2/rohit_gene_prediction/chest_testing/CheXpert-v1.0-small/train/"

    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        for study in os.listdir(item_path):
            file_folder_path = os.path.join(item_path, study)
            for root, dirs, files in os.walk(file_folder_path):
                for file_name in files:
                    input_path = os.path.join(root, file_name)
                    print(f"Processing file: {input_path}")
                    output_path = os.path.join(root, "processed"+file_name)
                    if output_path not in files:
                        enhanced, params = process_image_file(input_path, output_path, verbose=True)
                    else:
                        print(f"{output_path} is present so not processing" )
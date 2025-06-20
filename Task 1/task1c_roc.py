import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd

def analyze_masks(ground_truth_path, segmented_mask_path, visualize=False, default_threshold=0.5):
    """
    Analyzes the ground truth mask and segmented mask to compute TP, FP, TN, FN at a default threshold.
    Provides visualization of the masks and overlays if required.

    Parameters:
    - ground_truth_path (str): File path to the ground truth mask image.
    - segmented_mask_path (str): File path to the segmented mask image.
    - visualize (bool): If True, displays the masks and computed areas.
    - default_threshold (float): Threshold for binarizing the segmented mask for visualization.

    Returns:
    - ground_truth_flat (np.ndarray): Flattened ground truth labels (0 or 1).
    - segmented_scores_flat (np.ndarray): Flattened segmented mask scores (continuous values between 0 and 1).
    - metrics (dict): Dictionary containing TP, FP, TN, FN counts at the default threshold.
    """
    
    # Load the masks as grayscale images
    ground_truth = io.imread(ground_truth_path, as_gray=True)
    segmented_mask = io.imread(segmented_mask_path, as_gray=True)
    
    # Ensure that the segmented mask has continuous scores between 0 and 1
    if segmented_mask.max() > 1:
        segmented_mask = segmented_mask / 255.0
    
    # Binarize the ground truth mask: Assuming ground truth masks are binary with values 0 and 1
    ground_truth_bin = (ground_truth > 0.5).astype(np.uint8)
    
    # Binarize the segmented mask using the default threshold for visualization
    segmented_bin = (segmented_mask >= default_threshold).astype(np.uint8)
    
    # Compute True Positives (TP)
    TP = np.logical_and(ground_truth_bin, segmented_bin).sum()
    
    # Compute False Positives (FP)
    FP = np.logical_and(np.logical_not(ground_truth_bin), segmented_bin).sum()
    
    # Compute True Negatives (TN)
    TN = np.logical_and(np.logical_not(ground_truth_bin), np.logical_not(segmented_bin)).sum()
    
    # Compute False Negatives (FN)
    FN = np.logical_and(ground_truth_bin, np.logical_not(segmented_bin)).sum()
    
    # Adjusted ROI (union of ground truth and segmented masks)
    adjusted_ROI = np.logical_or(ground_truth_bin, segmented_bin).astype(np.uint8)
    
    metrics = {
        'True Positives': TP,
        'False Positives': FP,
        'True Negatives': TN,
        'False Negatives': FN
    }
    
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Ground Truth Mask
        axes[0,0].imshow(ground_truth_bin, cmap='gray')
        axes[0,0].set_title('Ground Truth Mask')
        axes[0,0].axis('off')
        
        # Segmented Mask
        axes[0,1].imshow(segmented_bin, cmap='gray')
        axes[0,1].set_title(f'Segmented Mask (Threshold={default_threshold})')
        axes[0,1].axis('off')
        
        # True Positives Overlay
        TP_mask = np.logical_and(ground_truth_bin, segmented_bin)
        axes[0,2].imshow(ground_truth_bin, cmap='gray')
        axes[0,2].imshow(TP_mask, cmap='jet', alpha=0.5)
        axes[0,2].set_title('True Positives')
        axes[0,2].axis('off')
        
        # False Positives Overlay
        FP_mask = np.logical_and(np.logical_not(ground_truth_bin), segmented_bin)
        axes[1,0].imshow(segmented_bin, cmap='gray')
        axes[1,0].imshow(FP_mask, cmap='autumn', alpha=0.5)
        axes[1,0].set_title('False Positives')
        axes[1,0].axis('off')
        
        # False Negatives Overlay
        FN_mask = np.logical_and(ground_truth_bin, np.logical_not(segmented_bin))
        axes[1,1].imshow(ground_truth_bin, cmap='gray')
        axes[1,1].imshow(FN_mask, cmap='winter', alpha=0.5)
        axes[1,1].set_title('False Negatives')
        axes[1,1].axis('off')
        
        # Adjusted ROI
        axes[1,2].imshow(adjusted_ROI, cmap='gray')
        axes[1,2].set_title('Adjusted ROI')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Flatten the masks to 1D arrays for ROC computation
    ground_truth_flat = ground_truth_bin.flatten()
    segmented_scores_flat = segmented_mask.flatten()
    
    return ground_truth_flat, segmented_scores_flat, metrics

def compute_roc_auc_manual(ground_truth, segmented_scores, plot=False):
    """
    Manually computes ROC curve and AUC score without using roc_curve from scikit-learn.
    Does NOT perform any thresholding beyond sorting the scores.

    Parameters:
    - ground_truth (np.ndarray): Flattened ground truth labels (0 or 1).
    - segmented_scores (np.ndarray): Flattened segmented mask scores (continuous values between 0 and 1).
    - plot (bool, optional): Whether to plot the ROC curve. Default is False.

    Returns:
    - fpr (np.ndarray): False Positive Rates.
    - tpr (np.ndarray): True Positive Rates.
    - auc_score (float): Area Under the ROC Curve.
    """
    
    # Sort scores and corresponding ground truth labels in descending order
    desc_score_indices = np.argsort(-segmented_scores)
    segmented_scores_sorted = segmented_scores[desc_score_indices]
    ground_truth_sorted = ground_truth[desc_score_indices]
    
    # Total positives and negatives
    P = np.sum(ground_truth_sorted)
    N = len(ground_truth_sorted) - P
    
    # Initialize TPR and FPR lists
    tpr = []
    fpr = []
    
    # Initialize counters
    TP = 0
    FP = 0
    
    # Iterate through sorted scores and compute TPR and FPR
    for i in range(len(segmented_scores_sorted)):
        if ground_truth_sorted[i] == 1:
            TP += 1
        else:
            FP += 1
        tpr.append(TP / P if P != 0 else 0)
        fpr.append(FP / N if N != 0 else 0)
    
    # Append (1,1) to ensure the ROC curve ends at (1,1)
    tpr.append(1.0)
    fpr.append(1.0)
    
    # Convert lists to numpy arrays
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    
    # Sort the FPR and TPR in ascending order of FPR to ensure proper plotting
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Calculate AUC using the trapezoidal rule
    auc_score = np.trapz(tpr_sorted, fpr_sorted)
    
    if plot:
        plt.figure(figsize=(8, 6))
        # Plot ROC Curve with markers
        plt.plot(fpr_sorted, tpr_sorted, color='blue', lw=2, marker='o', markersize=4,
                 label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random Classifier (AUC = 0.5)')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
    
    return fpr_sorted, tpr_sorted, auc_score

def process_directories(ground_truth_dir, segmented_masks_dir, output_txt_path, visualize=False, default_threshold=0.5):
    """
    Processes all mask pairs in the given directories, computes AUC for each,
    performs per-image ROC computation, and then plots TPR vs FPR as a scatter plot.

    Additionally computes and saves average IoU, Accuracy, and F1 Score metrics.

    Parameters:
    - ground_truth_dir (str): Directory containing ground truth mask images.
    - segmented_masks_dir (str): Directory containing segmented mask images.
    - output_txt_path (str): File path to save the AUC and additional metrics.
    - visualize (bool): If True, visualizes each mask pair.
    - default_threshold (float): Threshold for binarizing the segmented masks.

    Returns:
    - average_auc (float): Average AUC score across all images.
    - average_iou (float): Average IoU across all images.
    - average_accuracy (float): Average Accuracy across all images.
    - average_f1 (float): Average F1 Score across all images.
    """
    
    # Ensure directories exist
    if not os.path.isdir(ground_truth_dir):
        print(f"Ground truth directory does not exist: {ground_truth_dir}")
        return
    if not os.path.isdir(segmented_masks_dir):
        print(f"Segmented masks directory does not exist: {segmented_masks_dir}")
        return
    
    # List of ground truth files
    ground_truth_files = sorted([f for f in os.listdir(ground_truth_dir) if os.path.isfile(os.path.join(ground_truth_dir, f))])
    
    if not ground_truth_files:
        print(f"No ground truth mask files found in directory: {ground_truth_dir}")
        return
    
    # Initialize lists to store per-image metrics
    auc_scores = []
    fpr_points = []
    tpr_points = []
    iou_scores = []
    accuracy_scores = []
    f1_scores = []
    
    # Open the output text file
    with open(output_txt_path, 'w') as txt_file:
        # Write the header with additional columns for IoU, Accuracy, and F1 Score
        txt_file.write("Image_Name\tAUC_Score\tTrue_Positives\tFalse_Positives\tTrue_Negatives\tFalse_Negatives\tIoU\tAccuracy\tF1_Score\n")
        
        # Iterate over ground truth files
        for gt_file in ground_truth_files:
            gt_path = os.path.join(ground_truth_dir, gt_file)
            
            # Extract the base name of the ground truth file (without extension)
            base_name = os.path.splitext(gt_file)[0]
            
            # Now find a segmented mask that starts with base_name + '_'
            seg_candidates = [f for f in os.listdir(segmented_masks_dir) if f.startswith(base_name + "_")]
            
            if not seg_candidates:
                print(f"Segmented mask not found for image: {gt_file}. Skipping.")
                continue
            
            # Assume the first candidate is the correct segmented mask
            seg_file = seg_candidates[0]
            seg_path = os.path.join(segmented_masks_dir, seg_file)
            
            print(f"Processing image: {gt_file}")
            
            try:
                # Analyze masks
                ground_truth_flat, segmented_scores_flat, metrics = analyze_masks(
                    gt_path,
                    seg_path,
                    visualize=visualize,
                    default_threshold=default_threshold
                )
                
                # Compute ROC and AUC per image
                _, _, auc_score = compute_roc_auc_manual(
                    ground_truth_flat,
                    segmented_scores_flat,
                    plot=False  # Do not plot individual ROC curves
                )
                
                # Store AUC
                auc_scores.append(auc_score)
                
                # Extract TP, FP, TN, FN from metrics
                TP = metrics['True Positives']
                FP = metrics['False Positives']
                TN = metrics['True Negatives']
                FN = metrics['False Negatives']
                
                # Compute additional metrics
                iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
                accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
                f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
                
                # Store additional metrics
                iou_scores.append(iou)
                accuracy_scores.append(accuracy)
                f1_scores.append(f1_score)
                
                # Compute FPR and TPR at default threshold
                tpr = TP / (TP + FN) if (TP + FN) != 0 else 0
                fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
                tpr_points.append(tpr)
                fpr_points.append(fpr)
                
                # Write all metrics to text file
                txt_file.write(f"{gt_file}\t{auc_score:.4f}\t{TP}\t{FP}\t{TN}\t{FN}\t{iou:.4f}\t{accuracy:.4f}\t{f1_score:.4f}\n")
                
                # Print metrics
                print(f"AUC for {gt_file}: {auc_score:.4f}")
                print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
                print(f"IoU: {iou:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}\n")
            except Exception as e:
                print(f"Error processing image {gt_file}: {e}. Skipping.")
                continue
    
    # Compute average AUC across all images
    if auc_scores:
        average_auc = np.mean(auc_scores)
        print(f"Average AUC across all images: {average_auc:.4f}")
        with open(output_txt_path, 'a') as txt_file:
            txt_file.write(f"\nAverage_AUC\t{average_auc:.4f}\n")
    else:
        average_auc = 0.0
        print("No AUC scores were computed. Please check your mask directories and filenames.")
        with open(output_txt_path, 'a') as txt_file:
            txt_file.write(f"\nAverage_AUC\t{average_auc:.4f}\n")
    
    # Compute average IoU, Accuracy, and F1 Score across all images
    if iou_scores:
        average_iou = np.mean(iou_scores)
        average_accuracy = np.mean(accuracy_scores)
        average_f1 = np.mean(f1_scores)
        print(f"Average IoU across all images: {average_iou:.4f}")
        print(f"Average Accuracy across all images: {average_accuracy:.4f}")
        print(f"Average F1 Score across all images: {average_f1:.4f}")
        with open(output_txt_path, 'a') as txt_file:
            txt_file.write(f"Average_IoU\t{average_iou:.4f}\n")
            txt_file.write(f"Average_Accuracy\t{average_accuracy:.4f}\n")
            txt_file.write(f"Average_F1_Score\t{average_f1:.4f}\n")
    else:
        average_iou = 0.0
        average_accuracy = 0.0
        average_f1 = 0.0
        print("No IoU, Accuracy, or F1 Score metrics were computed.")
        with open(output_txt_path, 'a') as txt_file:
            txt_file.write(f"Average_IoU\t{average_iou:.4f}\n")
            txt_file.write(f"Average_Accuracy\t{average_accuracy:.4f}\n")
            txt_file.write(f"Average_F1_Score\t{average_f1:.4f}\n")
    
    # Create a scatter plot of TPR vs FPR for each image
    if fpr_points and tpr_points:
        print("Creating TPR vs FPR scatter plot...")
        
        plt.figure(figsize=(8, 6))
        plt.scatter(fpr_points, tpr_points, color='green', edgecolors='k', alpha=0.7)
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random Classifier (TPR=FPR)')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('TPR vs FPR Scatter Plot for Each Image')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Save the scatter plot with 300 DPI
        output_dir = os.path.dirname(output_txt_path)
        scatter_plot_filename = f"{output_dir} TPR_FPR_Scatter_Plot.png"
        scatter_plot_path = os.path.join(output_dir, scatter_plot_filename)
        plt.savefig(scatter_plot_path, dpi=300)
        print(f"TPR vs FPR scatter plot saved at: {scatter_plot_path}")
        
        plt.show()
    else:
        print("No data available to create TPR vs FPR scatter plot.")
    
    return average_auc, average_iou, average_accuracy, average_f1

# Example Usage
if __name__ == "__main__":
    ground_truth_dir = "./dataset/masks"
    segmented_masks_dir = "./task1_results/hsv_hough_contour_ellipse_masks" # Masks from Task1b run
    output_txt_path = "./task1_results/hsv_hough_contour_ellipse_masks.txt"

    # Process directories, compute per-image metrics, and create TPR vs FPR scatter plot
    average_auc, average_iou, average_accuracy, average_f1 = process_directories(
        ground_truth_dir,
        segmented_masks_dir,
        output_txt_path,
        visualize=False,
        default_threshold=0.5
    )
    
    # Display final results
    print("\nFinal Results:")
    print(f"Average AUC across all images: {average_auc:.4f}")
    print(f"Average IoU across all images: {average_iou:.4f}")
    print(f"Average Accuracy across all images: {average_accuracy:.4f}")
    print(f"Average F1 Score across all images: {average_f1:.4f}")
    print(f"AUC and additional metrics have been saved to: {output_txt_path}")

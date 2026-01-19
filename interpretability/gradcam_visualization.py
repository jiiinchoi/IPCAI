"""
Grad-CAM Visualization
TP 1개 + FN 1개 케이스 시각화
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models import ImageModel
from utils.metrics import set_seed, torch_load_weights


class GradCAM:
    """Grad-CAM for ResNet"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx):
        """Generate CAM for given class"""
        self.model.eval()
        
        # Forward
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward from target class
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU to keep positive influence
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_and_preprocess_image(nii_path, n_slices=6):
    """Load MRI and get center slices"""
    img = nib.load(nii_path).get_fdata()
    H, W, D = img.shape
    
    # Center slices
    center = D // 2
    start = max(0, center - n_slices // 2)
    end = min(D, start + n_slices)
    
    slices = img[:, :, start:end]
    
    # Pad if needed
    if slices.shape[2] < n_slices:
        pad_size = n_slices - slices.shape[2]
        slices = np.pad(slices, ((0, 0), (0, 0), (0, pad_size)), mode='constant')
    
    slices = slices.transpose(2, 0, 1)  # (n, H, W)
    
    # Resize to 128x128
    resized = []
    for i in range(slices.shape[0]):
        resized.append(cv2.resize(slices[i], (128, 128)))
    slices = np.stack(resized, axis=0)
    
    # Z-score normalize
    mean = slices.mean()
    std = slices.std()
    slices = (slices - mean) / (std + 1e-8)
    
    return slices.astype(np.float32)


def visualize_gradcam(original_slices, cam, title, save_path):
    """Visualize Grad-CAM heatmap overlaid on original slices"""
    n_slices = original_slices.shape[0]
    
    # Select middle slice for main visualization
    middle_idx = n_slices // 2
    original_slice = original_slices[middle_idx]
    
    # Normalize original for display
    original_slice = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min() + 1e-8)
    
    # Resize CAM to match original
    cam_resized = cv2.resize(cam, (128, 128))
    
    # Create heatmap
    heatmap = plt.cm.jet(cam_resized)[:, :, :3]  # RGB
    
    # Overlay
    overlay = 0.6 * original_slice[:, :, np.newaxis] + 0.4 * heatmap
    overlay = overlay / overlay.max()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title('Original MRI Slice', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (High activation = Red)', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    set_seed(42)
    device = get_device()
    n_slices = 6
    
    print("\n" + "="*70)
    print("  Grad-CAM Visualization")
    print("="*70)
    print(f"Condition: N={n_slices} slices\n")
    
    # Load model
    img_ckpt = ROOT / "runs/image_t1ce/models/best_image.pt"
    model = ImageModel(num_classes=2, pretrained=True).to(device)
    model.load_state_dict(torch_load_weights(img_ckpt, device))
    model.eval()
    
    # Grad-CAM on last conv layer of ResNet
    target_layer = model.resnet.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    print("✅ Model and Grad-CAM ready\n")
    
    # Load test data
    df = pd.read_csv(ROOT / 'data/splits/test.csv')
    
    # Get predictions for all test cases
    print("Getting predictions for all test cases...")
    predictions = []
    
    for idx, row in df.iterrows():
        nii_path = ROOT / row['t1ce']
        label = int(row['label'])
        
        # Load image
        slices = load_and_preprocess_image(str(nii_path), n_slices)
        input_tensor = torch.from_numpy(slices).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].item()
            pred = 1 if prob > 0.5 else 0
        
        predictions.append({
            'idx': idx,
            'path': nii_path,
            'label': label,
            'pred': pred,
            'prob': prob,
            'correct': pred == label
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")
    
    # Find cases
    tp_cases = [p for p in predictions if p['label'] == 1 and p['pred'] == 1]
    fn_cases = [p for p in predictions if p['label'] == 1 and p['pred'] == 0]
    
    print(f"\nFound {len(tp_cases)} TP cases and {len(fn_cases)} FN cases")
    
    if len(tp_cases) == 0 or len(fn_cases) == 0:
        print("⚠️ Not enough cases for visualization")
        return
    
    # Select cases
    # TP: highest confidence
    tp_case = max(tp_cases, key=lambda x: x['prob'])
    # FN: closest to threshold (most interesting)
    fn_case = max(fn_cases, key=lambda x: x['prob'])
    
    print(f"\nSelected cases:")
    print(f"  TP: Case #{tp_case['idx']}, prob={tp_case['prob']:.3f}")
    print(f"  FN: Case #{fn_case['idx']}, prob={fn_case['prob']:.3f}")
    
    # Generate Grad-CAMs
    output_dir = ROOT / 'results/interpretability'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for case_type, case in [('TP', tp_case), ('FN', fn_case)]:
        print(f"\nGenerating Grad-CAM for {case_type}...")
        
        # Load image
        slices = load_and_preprocess_image(str(case['path']), n_slices)
        input_tensor = torch.from_numpy(slices).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        # Generate CAM for positive class (class 1)
        cam = grad_cam.generate_cam(input_tensor, class_idx=1)
        
        # Visualize
        title = f"{case_type} Case #{case['idx']}: Label={case['label']}, Pred={case['pred']} (p={case['prob']:.3f})"
        save_path = output_dir / f'gradcam_{case_type.lower()}_case{case['idx']}.png'
        
        visualize_gradcam(slices, cam, title, save_path)
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
For TP (True Positive):
→ Red regions show where model detected ET presence
→ High activation should correlate with tumor location
→ Demonstrates model learned relevant features

For FN (False Negative):
→ Shows what model focused on (incorrectly)
→ May reveal subtle features or artifacts
→ Helps understand failure modes
→ Low/diffuse activation suggests weak signal

Clinical Insight:
→ If TP shows focal activation → model uses localized features
→ If FN shows weak activation → insufficient evidence (low quality)
→ Validates that N=6 is challenging even for trained model
    """)
    
    print("="*70)
    print("\n✅ Grad-CAM visualization completed!")


if __name__ == "__main__":
    main()
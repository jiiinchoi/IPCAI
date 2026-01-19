"""
Models for BraTS Classification
- ImageModel: ResNet-18 기반
- TextModel: ClinicalBERT 기반
- MultimodalModel: Late Fusion
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel


class ImageModel(nn.Module):
    """
    ResNet-18 기반 Image Classification
    
    입력: (B, N, H, W) - N개 슬라이스
    출력: (B, 2) - 2-class logits
    
    처리:
      1) 각 슬라이스를 ResNet에 통과 -> (B*N, 2)
      2) 슬라이스별 logits를 평균 -> (B, 2)
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # ResNet-18 backbone
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
        
        # 1채널 입력으로 변경 (grayscale MRI)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # FC layer 교체
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, H, W) - N slices per case
        
        Returns:
            logits: (B, 2)
        """
        B, N, H, W = x.shape
        
        # Reshape: (B, N, H, W) -> (B*N, 1, H, W)
        x = x.reshape(B * N, 1, H, W)
        
        # Forward through ResNet
        logits = self.backbone(x)  # (B*N, 2)
        
        # Reshape back and average
        logits = logits.reshape(B, N, -1)  # (B, N, 2)
        logits = logits.mean(dim=1)         # (B, 2)
        
        return logits


class TextModel(nn.Module):
    """
    ClinicalBERT 기반 Text Classification
    
    입력: input_ids, attention_mask
    출력: (B, 2) - 2-class logits
    """
    
    def __init__(
        self,
        num_classes=2,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        freeze_bert=False
    ):
        super().__init__()
        
        # BERT 모델 로드 (safetensors 사용)
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        
        # BERT freeze 옵션
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        
        Returns:
            logits: (B, 2)
        """
        # BERT forward
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        
        # Classification
        logits = self.classifier(cls_embedding)  # (B, 2)
        
        return logits


class MultimodalModel(nn.Module):
    """
    Late Fusion Multimodal Model
    
    Fusion modes:
      - 'weighted': p = alpha * p_img + (1-alpha) * p_txt
      - 'learned': concat logits -> linear -> fused logits
    """
    
    def __init__(
        self,
        image_model: nn.Module,
        text_model: nn.Module,
        fusion_mode: str = 'weighted',
        alpha: float = 0.5
    ):
        super().__init__()
        
        self.image_model = image_model
        self.text_model = text_model
        self.fusion_mode = fusion_mode
        self.alpha = float(alpha)
        
        # Learned fusion layer
        if fusion_mode == 'learned':
            # [img_logits(2), txt_logits(2)] -> fused_logits(2)
            self.fusion_layer = nn.Linear(4, 2)
        else:
            self.fusion_layer = None
    
    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images: (B, N, H, W)
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        
        Returns:
            fused_logits: (B, 2)
            img_logits: (B, 2)
            txt_logits: (B, 2)
        """
        # Forward through each branch
        img_logits = self.image_model(images)
        txt_logits = self.text_model(input_ids, attention_mask)
        
        # Fusion
        if self.fusion_mode == 'weighted':
            # Probability-level fusion
            img_probs = torch.softmax(img_logits, dim=1)
            txt_probs = torch.softmax(txt_logits, dim=1)
            
            fused_probs = self.alpha * img_probs + (1 - self.alpha) * txt_probs
            
            # Convert back to logits
            fused_logits = torch.log(fused_probs + 1e-8)
        
        elif self.fusion_mode == 'learned':
            # Logit-level fusion
            combined = torch.cat([img_logits, txt_logits], dim=1)  # (B, 4)
            fused_logits = self.fusion_layer(combined)              # (B, 2)
        
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        
        return fused_logits, img_logits, txt_logits


# 테스트 코드
if __name__ == "__main__":
    print("==== Models 테스트 ====\n")
    
    # 더미 데이터
    B, N, H, W = 4, 16, 128, 128
    seq_len = 512
    
    dummy_images = torch.randn(B, N, H, W)
    dummy_input_ids = torch.randint(0, 1000, (B, seq_len))
    dummy_attention_mask = torch.ones(B, seq_len)
    
    print(f"Dummy data shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Input IDs: {dummy_input_ids.shape}")
    print(f"  Attention Mask: {dummy_attention_mask.shape}\n")
    
    # Image Model
    print("[1] ImageModel")
    try:
        img_model = ImageModel(num_classes=2, pretrained=False)
        img_logits = img_model(dummy_images)
        print(f"  Output shape: {img_logits.shape}")
        print(f"  ✅ ImageModel works!\n")
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
    
    # Text Model (실제 환경에서만 작동)
    print("[2] TextModel")
    print("  (HuggingFace 모델 로드 필요 - 실제 환경에서 테스트)\n")
    
    # Multimodal
    print("[3] MultimodalModel")
    print("  (실제 환경에서 테스트)")
    
    print("\n✅ Models 코드 작성 완료!")
    print("\n사용 예시:")
    print("""
    # Image-only
    img_model = ImageModel(num_classes=2, pretrained=True)
    logits = img_model(images)  # (B, 2)
    
    # Text-only
    txt_model = TextModel(num_classes=2, freeze_bert=True)
    logits = txt_model(input_ids, attention_mask)  # (B, 2)
    
    # Multimodal
    mm_model = MultimodalModel(img_model, txt_model, fusion_mode='weighted', alpha=0.5)
    fused_logits, img_logits, txt_logits = mm_model(images, input_ids, attention_mask)
    """)
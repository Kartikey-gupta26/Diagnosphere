import torch
import torchvision
from torchvision import models
from transformers import ViTForImageClassification
from tensorflow.keras.models import load_model
from pathlib import Path
import logging
from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        """Initialize model loader and load all models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(__file__).parent / 'trained_models'
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load all trained models into memory with error handling"""
        try:
            logger.info("Starting model loading process...")
            
            # Base model
            logger.info("Loading base model...")
            self.base_model = models.efficientnet_b0(weights=None)
            self.base_model.classifier[1] = torch.nn.Linear(
                self.base_model.classifier[1].in_features, 
                3
            )
            self._load_model_weights(
                self.base_model, 
                'base_model.pth',
                'base model'
            )
            
            # Brain model
            logger.info("Loading brain model...")
            self.brain_model = models.mobilenet_v2(weights=None)
            self.brain_model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(self.brain_model.last_channel, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
                torch.nn.Sigmoid()
            )
            self._load_model_weights(
                self.brain_model, 
                'brain_model.pth',
                'brain model'
            )
            
            # Lungs model
            logger.info("Loading lungs model...")
            lungs_model_path = self.model_dir / 'lungs_model.h5'
            if not lungs_model_path.exists():
                raise FileNotFoundError(f"Lungs model not found at {lungs_model_path}")
            self.lungs_model = load_model(str(lungs_model_path))
            
            # Skin model
            logger.info("Loading skin model...")
            self.skin_model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k',
                num_labels=23
            )
            self._load_model_weights(
                self.skin_model,
                'skin_model.pth',
                'skin model'
            )
            
            # Common image transform
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.models_loaded = True
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            self.models_loaded = False
            raise
    
    def _load_model_weights(self, model, weight_file, model_name):
        """Helper method to load model weights with error handling"""
        weight_path = self.model_dir / weight_file
        if not weight_path.exists():
            raise FileNotFoundError(
                f"{model_name} weights not found at {weight_path}"
            )
        
        try:
            state_dict = torch.load(
                str(weight_path),
                map_location=self.device
            )
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            logger.info(f"Successfully loaded {model_name} weights")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_name} weights: {str(e)}"
            ) from e
    
    def verify_models(self):
        """Verify all required models are loaded"""
        required_models = [
            'base_model',
            'brain_model', 
            'lungs_model',
            'skin_model'
        ]
        
        missing_models = [
            name for name in required_models 
            if not hasattr(self, name) or getattr(self, name) is None
        ]
        
        if missing_models:
            raise RuntimeError(
                f"The following models failed to load: {', '.join(missing_models)}"
            )
        return True
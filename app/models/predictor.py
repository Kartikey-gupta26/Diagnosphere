import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import tensorflow as tf

class DiagnoSpherePredictor:
    def __init__(self, model_loader):
        self.ml = model_loader
        self.class_names = {
            'base': ['brain', 'lungs', 'skin'],
            'brain': ['No Tumor', 'Tumor'],
            'lungs': ['BacterialPneumonia', 'COVID-19', 'Normal', 'ViralPneumonia'],
            'skin': ['Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 'Candidiasis', 
         'DrugEruption', 'Eczema', 'Infestations_Bites', 'Lichen', 'Lupus', 
         'Moles', 'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'SkinCancer', 
         'Sun_Sunlight_Damage', 'Tinea', 'Unknown_Normal', 'Vascular_Tumors', 
         'Vasculitis', 'Vitiligo', 'Warts']

        }
    
    def preprocess_image(self, image_path, framework='pytorch'):
        """Handle preprocessing for different frameworks"""
        if framework == 'pytorch':
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(image).unsqueeze(0).to(self.ml.device)
        else:  # tensorflow
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(256, 256))
            return tf.keras.preprocessing.image.img_to_array(img) / 255.0
    
    def predict(self, image_path):
        try:
            # First determine domain
            image_tensor = self.preprocess_image(image_path)
            with torch.no_grad():
                base_output = self.ml.base_model(image_tensor)
                domain_prob = torch.nn.functional.softmax(base_output, dim=1)
                domain = torch.argmax(domain_prob).item()
            
            result = {
                'domain': self.class_names['base'][domain],
                'domain_confidence': float(domain_prob[0][domain]),
                'diagnosis': None,
                'diagnosis_confidence': None
            }
            
            # Route to specialized model
            if domain == 0:  # Brain
                with torch.no_grad():
                    output = self.ml.brain_model(image_tensor)
                    result['diagnosis'] = self.class_names['brain'][int(output > 0.5)]
                    result['diagnosis_confidence'] = float(output if output > 0.5 else 1 - output)
            
            elif domain == 1:  # Lungs
                img_array = np.expand_dims(self.preprocess_image(image_path, 'tensorflow'), axis=0)
                predictions = self.ml.lungs_model.predict(img_array)
                diagnosis = np.argmax(predictions[0])
                result['diagnosis'] = self.class_names['lungs'][diagnosis]
                result['diagnosis_confidence'] = float(predictions[0][diagnosis])
            
            else:  # Skin
                with torch.no_grad():
                    outputs = self.ml.skin_model(image_tensor).logits
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    diagnosis = torch.argmax(probs).item()
                    result['diagnosis'] = self.class_names['skin'][diagnosis]
                    result['diagnosis_confidence'] = float(probs[0][diagnosis])
            
            return {'status': 'success', 'result': result}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
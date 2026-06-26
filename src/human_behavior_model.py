import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv; load_dotenv(); acceptance_threshold = float(os.getenv("ACCEPTANCE_THRESHOLD", 0.50))

class ACCEPTANCE_MODEL(nn.Module):
    def __init__(self, input_size):
        super(ACCEPTANCE_MODEL, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)  
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  
        return x
    
class G_FUNCTION(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_a = nn.Parameter(torch.randn(1))
        self.raw_b = nn.Parameter(torch.randn(1))
    
    def forward(self, A):
        A = A.float()
        a = self.raw_a
        b = self.raw_b
        sign_A = torch.sign(A)
        exponent = -sign_A * (a * torch.abs(A) + b)
        return 1 / (1 + torch.exp(exponent))

class HUMAN_BEHAVIOR_MODEL:
    def __init__(self):      
        self.a_model = self._load_model("./hb_models/acceptance_model.pth")
        self.g_model = self._load_model("./hb_models/g_model.pth")
        
    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        state_dict = torch.load(path, map_location='cpu')

        if 'acceptance_model' in path:
            model = ACCEPTANCE_MODEL(input_size=10)
            model.load_state_dict(state_dict)
        else:
            model = G_FUNCTION()
            
            if isinstance(state_dict, dict) and 'raw_a' in state_dict and 'raw_b' in state_dict:
                if isinstance(state_dict['raw_a'], (float, int)):
                    state_dict['raw_a'] = torch.tensor([state_dict['raw_a']])
                elif state_dict['raw_a'].dim() == 0: 
                    state_dict['raw_a'] = state_dict['raw_a'].unsqueeze(0)
                
                if isinstance(state_dict['raw_b'], (float, int)):
                    state_dict['raw_b'] = torch.tensor([state_dict['raw_b']])
                elif state_dict['raw_b'].dim() == 0:  
                    state_dict['raw_b'] = state_dict['raw_b'].unsqueeze(0)
            
            model.load_state_dict(state_dict)
        
        model.eval()
        return model
    
    def get_prediction(self, confidence: float, model_type: Literal['a', 'g'], user_features: dict = None, measure: int = None) -> float:
        if model_type == 'g':
            model = self.g_model
            with torch.no_grad():
                print(f"Confidence input to G model: {confidence}")
                input_tensor = torch.tensor([confidence], dtype=torch.float32)
                prediction = model(input_tensor)
                print(f"Prediction from G model: {prediction.item()}")
                return prediction.item()
        
        elif model_type == 'a':
            if user_features is None:
                raise ValueError("user_features required for a_model")
            
            model = self.a_model
            features = [
                user_features['age'],
                user_features['gender'],
                user_features['socioeconomic_status'],
                user_features['survey_q4_prefers_human_or_AI'],
                user_features['survey_q5_AI_in_life'],
                user_features['education'],
                user_features['programming_experience'],
                measure,
                confidence,
                measure * confidence
            ]
            
            with torch.no_grad():
                input_tensor = torch.tensor(features, dtype=torch.float32)
                prediction = model(input_tensor)
                return prediction.item()
    
    @staticmethod
    def pick_random_demographic(df):
        df = df[(df['task_name'] == 'art') & (df['geographic_region'] == 'United States') & (df['perceived_accuracy'] == 80)]
        df = df[df['gender'] != 'prefer not to say']
        df['gender'] = df['gender'].map({'male': 0, 'female': 1})
        df['programming_experience'] = df['programming_experience'].map({'No': 0, "I don't know": 0, 'Yes': 1})
        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()
        df[['socioeconomic_status', 'education']] = minmax_scaler.fit_transform(df[['socioeconomic_status', 'education']])
        df[['age']] = standard_scaler.fit_transform(df[['age']])
        row = df.sample(n=1).iloc[0]
        return {
            'age': row['age'],
            'gender': row['gender'],
            'socioeconomic_status': row['socioeconomic_status'],
            'survey_q4_prefers_human_or_AI': row['survey_q4_prefers_human_or_AI'],
            'survey_q5_AI_in_life': row['survey_q5_AI_in_life'],
            'education': row['education'],
            'programming_experience': row['programming_experience'],
        }
    
    @staticmethod
    def confirm_user_acceptability(acceptance_score:float) -> bool:
        return acceptance_score > float(acceptance_threshold)
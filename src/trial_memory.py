from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv; load_dotenv(); max_reflexion_trials = int(os.getenv("MAX_REFLEXION_TRIALS", 10))

@dataclass 
class TRIAL:
    trial_id: int
    timestamp: datetime
    episode: int
    question: str
    actor_advice: str
    evaluator_correctness: float
    evaluator_acceptability: bool
    g_model_output: float
    activation_model_output: float
    reflexion_output: str
    loss: Optional[float] = None
    actor_confidence: Optional[float] = None
    g_input: Optional[float] = None
    user_acceptability_confirmed: Optional[bool] = None
    food_agreement: Optional[int] = None
    transportation_agreement: Optional[int] = None
    accommodation_agreement: Optional[int] = None
    activities_agreement: Optional[int] = None
    shopping_agreement: Optional[int] = None
    nightlife_agreement: Optional[int] = None
    budget_agreement: Optional[int] = None
    overall_agreement: Optional[bool] = None
    question_id: Optional[int] = None

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "iteration": self.episode,
            "timestamp": self.timestamp.isoformat(),
            "question": self.question,
            "actor_advice": self.actor_advice,
            "evaluator_correctness": self.evaluator_correctness,
            "evaluator_acceptability": self.evaluator_acceptability,
            "g_model_output": self.g_model_output,
            "activation_model_output": self.activation_model_output,
            "reflexion_output": self.reflexion_output,
            "loss": self.loss,
            "actor_confidence": self.actor_confidence,
            "g_input": self.g_input,
            "user_acceptability_confirmed": self.user_acceptability_confirmed,
            "food_agreement": self.food_agreement,
            "transportation_agreement": self.transportation_agreement,
            "accommodation_agreement": self.accommodation_agreement,
            "activities_agreement": self.activities_agreement,
            "shopping_agreement": self.shopping_agreement,
            "nightlife_agreement": self.nightlife_agreement,
            "budget_agreement": self.budget_agreement,
            "overall_agreement": self.overall_agreement,
        }

@dataclass
class TRIAL_MEMORY:
    trials: List[TRIAL] = field(default_factory=list)
    max_trials: int = int(max_reflexion_trials)
    _next_trial_id: int = 1
    
    def add_trial(self, question: str, episode: int, actor_advice: str, evaluator_correctness: float, 
                  evaluator_acceptability: bool, g_model_output: float, activation_model_output: float, 
                  reflexion_output: str, actor_confidence: Optional[float] = None,
                  g_input: Optional[float] = None, user_acceptability_confirmed: Optional[bool] = None,
                  food_agreement: Optional[int] = None, transportation_agreement: Optional[int] = None,
                  accommodation_agreement: Optional[int] = None, activities_agreement: Optional[int] = None,
                  shopping_agreement: Optional[int] = None, nightlife_agreement: Optional[int] = None,
                  budget_agreement: Optional[int] = None, overall_agreement: Optional[bool] = None, question_id: Optional[int] = None,
                  loss:float = None) -> TRIAL:
        if len(self.trials) > self.max_trials:
            raise ValueError(f'Maximum number of trials {self.max_trials} exceeded.')
        
        trial = TRIAL(
            trial_id=self._next_trial_id,
            timestamp=datetime.now(),
            question_id=question_id,
            episode=episode,
            question=question,
            actor_advice=actor_advice,
            evaluator_correctness=evaluator_correctness,
            evaluator_acceptability=evaluator_acceptability,
            g_model_output=g_model_output,
            activation_model_output=activation_model_output,
            reflexion_output=reflexion_output,
            loss=loss,
            actor_confidence=actor_confidence,
            g_input=g_input,
            user_acceptability_confirmed=user_acceptability_confirmed,
            food_agreement=food_agreement,
            transportation_agreement=transportation_agreement,
            accommodation_agreement=accommodation_agreement,
            activities_agreement=activities_agreement,
            shopping_agreement=shopping_agreement,
            nightlife_agreement=nightlife_agreement,
            budget_agreement=budget_agreement,
            overall_agreement=overall_agreement,
        )
        
        self.trials.append(trial)
        self._next_trial_id += 1
            
        return trial
    
    def get_recent_trials(self, count: int = 3) -> List[TRIAL]:
        return self.trials[-count:]
    
    def get_reflexion_history(self, count: int = 3) -> str:
        recent = self.get_recent_trials(count)
        history = ""
        
        for trial in recent:
            history += f"Trial {trial.trial_id}:\n"
            history += f"Question: {trial.question}\n"
            history += f"Actor: {trial.actor_advice}\n"
            history += f"Correctness: {trial.evaluator_correctness:.2f}\n"
            history += f"Reflexion: {trial.reflexion_output}\n\n"
            
        return history.strip()
    

class LONG_TERM_MEMORY:
    def __init__(self):
        self.trial_memories: Dict[int, TRIAL_MEMORY] = {}
        self._next_id: int = 1
    
    def store_trial_memory(self, trial_memory: TRIAL_MEMORY) -> int:
        memory_id = self._next_id
        self.trial_memories[memory_id] = trial_memory
        self._next_id += 1
        return memory_id

    def to_dict(self):
        all_trials = []
        for memory in self.trial_memories.values():
            for trial in memory.trials:
                all_trials.append(trial.to_dict())
        return all_trials

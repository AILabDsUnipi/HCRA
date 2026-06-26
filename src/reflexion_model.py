from langchain_core.prompts import PromptTemplate
from llm_caller import DeepSeekCaller, DeepSeekConfig
from trial_memory import TRIAL_MEMORY
import os
from dotenv import load_dotenv; load_dotenv(); reflexion_temp = float(os.getenv("REFLEXION_TEMP", 1.0))

class REFLEXION:
    def __init__(self):
        self.llm = DeepSeekCaller(
            config=DeepSeekConfig(
                temperature=reflexion_temp,
                max_tokens=300
            )
        )
        
        self.reflexion_template = PromptTemplate(
            input_variables=["trial_history", "latest_actor_answer", "is_correct", "user_acceptability"],
            template="""
                You are a reflexion agent analyzing an actor's performance. Review the trial history and provide constructive feedback.
                All questions have to do with turism destinations in Athens, Greece.
                The current evaluation is given by 2 trained models and should be considered as a fact.

                Trial History:
                {trial_history}

                Latest Actor Answer: {latest_actor_answer}

                Current Evaluation:
                - Answer Correctness: {is_correct}
                - User Acceptability: {user_acceptability:.2f}

                Based on the patterns in past answers, confidences, and outcomes, provide feedback to improve the actor's:
                1. Answer accuracy
                2. Confidence calibration 
                3. Predicted user acceptance of the answer

                Focus on actionable insights. Be concise but thorough. Don't give exact examples of answers.
            """
        )

    def generate_reflexion(self, trial_memory: TRIAL_MEMORY, latest_actor_answer: str, is_correct: bool, user_acceptability: float, long_term_memory=None) -> str:
        history = self._format_trial_history(trial_memory, long_term_memory)

        prompt = self.reflexion_template.format(trial_history=history, latest_actor_answer=latest_actor_answer, is_correct=is_correct, user_acceptability=user_acceptability)
        
        response = self.llm.call(prompt)
        return response.content
    
    def _format_trial_history(self, trial_memory: TRIAL_MEMORY, long_term_memory=None) -> str:
        history = ""

        if long_term_memory:
            history += "Previous Questions Context:\n"
            for memory_id, stored_memory in long_term_memory.trial_memories.items():
                if stored_memory.trials:
                    first_trial = stored_memory.trials[0]
                    last_trial = stored_memory.trials[-1]
                    history += f"Question: {first_trial.question}\n"
                    history += f"Final Answer: {last_trial.actor_advice}\n"
                    history += f"Final Reflexion: {last_trial.reflexion_output}\n\n"
            history += "Current Question Trials:\n"
        
        if not trial_memory.trials:
            history += "No previous trials for current question.\n"
        else:
            for trial in trial_memory.trials:
                history += f"Episode {trial.episode} (Trial {trial.trial_id}):\n" 
                history += f"Question: {trial.question}\n"
                history += f"Actor Answer: {trial.actor_advice}\n"
                try:
                    confidence_str = f"{trial.actor_confidence:.2f}"
                except:
                    confidence_str = "N/A"
                history += f"Actor Confidence: {confidence_str}\n"
                history += f"Was accepted by the evaluator (is the answer by the actor factually correct): {trial.evaluator_acceptability}\n"
                history += f"Trial Reflexion: {trial.reflexion_output}\n\n"
        
        return history.strip()
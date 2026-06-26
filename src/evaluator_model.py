from typing import Tuple, Union, Dict, Any
from langchain_core.prompts import PromptTemplate
from llm_caller import DeepSeekCaller, DeepSeekConfig
import os
from dotenv import load_dotenv; load_dotenv(); evaluator_temp = float(os.getenv("EVALUATOR_TEMP", 0.0))

class EVALUATOR:
    def __init__(self, correctness_threshold: float = 0.7):
        self.llm = DeepSeekCaller(
            config=DeepSeekConfig(
                temperature=evaluator_temp,
                max_tokens=150
            )
        )
        self.correctness_threshold = correctness_threshold

    def _get_agreement_scores(self, question: str, given_answer: str, user_preferences: Dict[str, Any], relevant_params: list[str]) -> Dict[str, int]:
        if not relevant_params:
            return {}

        params_with_prefs = [param for param in user_preferences.keys() if user_preferences[param] is not None]
        
        if not params_with_prefs:
            return {}

        agreement_prompts = []
        for param in params_with_prefs:
            agreement_prompts.append(f"- {param.capitalize()}_Agreement: Does the answer align with the user\'s {param} preference? (-1 for poor alignment, 1 for good alignment)")

        user_prefs_str = "\n".join([f"- {key.capitalize()}: {value}" for key, value in user_preferences.items() if key in params_with_prefs])

        eval_prompt = PromptTemplate(
            input_variables=["question", "given_answer", "user_preferences", "agreement_prompts"],
            template=
            '''
            You are an expert on Athens tourism. For the question: {question}
            The given answer is: {given_answer}

            User preferences:
            {user_preferences}

            Based on the user preferences, predict how well the answer aligns on the following dimensions:
            {agreement_prompts}

            Respond ONLY with:
            [Parameter]_Agreement: [-1 or 1]
            for each of the requested parameters. For example:
            Food_Agreement: 1
            Budget_Agreement: -1
            '''
        )

        formatted_prompt = eval_prompt.format(
            question=question,
            given_answer=given_answer,
            user_preferences=user_prefs_str,
            agreement_prompts="\n".join(agreement_prompts)
        )

        llm_response = self.llm.call(formatted_prompt)
        
        individual_agreements = {}
        try:
            lines = llm_response.content.split('\n')
            for param in params_with_prefs:
                param_key = f'{param}_agreement:'.lower()
                found_line = None
                for line in lines:
                    if param_key in line.lower():
                        found_line = line
                        break
                
                if found_line:
                    agreement_score = int(found_line.split(':')[1].strip())
                    individual_agreements[param] = agreement_score
                else:
                    individual_agreements[param] = -1
        except Exception as e:
            print(f'Error parsing agreement parameters for question: "{question}". Error: {e}. Using default values.')
            for param in params_with_prefs:
                individual_agreements[param] = -1

        return individual_agreements

    def evaluate(self, question: str, given_answer: str, agreement_parameters: bool = False, user_preferences: Dict[str, Any] = None, relevant_params: list[str] = None) -> Union[Tuple[float, bool], Tuple[float, bool, bool, Dict[str, int]]]:
        correctness_prompt = PromptTemplate(
            input_variables=["question", "given_answer"],
            template=
            '''
            You are an expert on Athens tourism. For the question: {question}
            The given answer is: {given_answer}
                    
            Rate the correctness of this answer on a scale of 0.0 to 1.0 based on factual accuracy.
                    
            Respond ONLY with:
            Correctness: [0.0-1.0 numerical score]
            '''
        )
        formatted_prompt = correctness_prompt.format(question=question, given_answer=given_answer)
        llm_response = self.llm.call(formatted_prompt)
        
        try:
            score_line = [line for line in llm_response.content.split('\n') if 'correctness:' in line.lower()][0]
            correctness_score = float(score_line.split(':')[1].strip())
        except:
            print('Error parsing correctness score from Evaluator LLM response. Using default value of 0.0.')
            correctness_score = 0.0
       
        is_correct = correctness_score >= self.correctness_threshold
        
        if not agreement_parameters:
            return correctness_score, is_correct
        
        if relevant_params is None:
            relevant_params = [param for param in user_preferences.keys() if user_preferences[param] is not None]

        if not relevant_params:
            return correctness_score, is_correct, True, {}

        individual_agreements = self._get_agreement_scores(question, given_answer, user_preferences, relevant_params)
        
        if not individual_agreements:
            overall_agreement = True
        else:
            overall_agreement = all(score == 1 for score in individual_agreements.values())

        return correctness_score, is_correct, overall_agreement, individual_agreements
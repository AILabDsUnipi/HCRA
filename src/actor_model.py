from langchain_core.prompts import PromptTemplate
from llm_caller import DeepSeekCaller, DeepSeekConfig
from trial_memory import TRIAL_MEMORY
import os
from dotenv import load_dotenv; load_dotenv(); actor_temp = float(os.getenv("ACTOR_TEMP", 1.0))

class ACTOR:
   def __init__(self):
       self.actor_llm = DeepSeekCaller(
           config=DeepSeekConfig(
               temperature=actor_temp,
               max_tokens=150
           )
       )
       self.task_template: str = """
       <|begin_of_text|>
       <|start_header_id|>system<|end_header_id|>
           You are an actor in a reflexion environment. This is the context you are given. Under every answer were your previous answers and under every reflection were every reflection from an agent that knows the correct answer for the question (prompt) at hand.
           {memory}
       <|eot_id|>
       <|start_header_id|>question<|end_header_id|>
           Taking the current context into account, what would you answer to this? Answer in about 50 words but do not mention how many words the answer was. Keep your answers VERY precise. Don't talk about the context and go straight to the answer. Never say anything about the reflective process but take into account all of your reflections.
           {prompt}
           After your answer, on a new line, rate your confidence as a single number between 0.0 and 1.0. This score should reflect how confident you feel about your answer overall. A score of 1.0 means you are completely confident. A score of 0.0 means you are not confident at all.
           Write exactly:
           Confidence: [0.0-1.0]
       <|eot_id|>
       assistant
       """
       self.prompt_wrapper: PromptTemplate = PromptTemplate(
           input_variables=["memory", "prompt"],
           template=self.task_template
       )
      
   def respond(self, trial_memory: TRIAL_MEMORY, prompt: str) -> tuple[str, float]:
       formatted_memory = trial_memory.get_reflexion_history()
       
       formatted_prompt = self.prompt_wrapper.format(
           memory=formatted_memory,
           prompt=prompt
       )
      
       response = self.actor_llm.call(formatted_prompt)

       content = response.content
       confidence = 0.5
       
       lines = content.strip().split('\n')
       for line in lines:
            if line.lower().startswith('confidence:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    pass
       clean_lines = [l for l in lines if not l.lower().startswith('confidence:')]
       clean_content = '\n'.join(clean_lines).strip()
      
       return clean_content, confidence
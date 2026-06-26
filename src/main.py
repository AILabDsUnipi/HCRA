import os
import random
import pandas as pd
import logging
import json
import concurrent.futures
from datetime import datetime
from actor_model import ACTOR
from evaluator_model import EVALUATOR
from reflexion_model import REFLEXION
from human_behavior_model import HUMAN_BEHAVIOR_MODEL
from trial_memory import TRIAL_MEMORY, LONG_TERM_MEMORY
from dotenv import load_dotenv; load_dotenv(); max_reflexion_trials = int(os.getenv("MAX_REFLEXION_TRIALS", 10))
from stopping_rules import stopping_rule, loss_function 
from wakepy import keep
import time

def setup_logging(trial_id: int):
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   log_filename = f"hcra_run{trial_id}_{timestamp}.txt"
   
   os.makedirs("logs", exist_ok=True)
   log_filepath = os.path.join("logs", log_filename)

   logger = logging.getLogger(f"run_{trial_id}")
   logger.setLevel(logging.INFO)

   logger.handlers.clear()

   file_handler = logging.FileHandler(log_filepath)
   console_handler = logging.StreamHandler()
   
   formatter = logging.Formatter('%(message)s')
   file_handler.setFormatter(formatter)
   console_handler.setFormatter(formatter)
   
   logger.addHandler(file_handler)
   logger.addHandler(console_handler)

   logging.getLogger("urllib3").setLevel(logging.WARNING)
   logging.getLogger("requests").setLevel(logging.WARNING)
   logging.getLogger("httpx").setLevel(logging.WARNING)
   logging.getLogger("httpcore").setLevel(logging.WARNING)
   
   return log_filepath, logger

def get_user_preferences(question: str, df: pd.DataFrame):
    question_row = df[df['Questions'] == question]

    if question_row.empty:
        return {}
    
    row = question_row.iloc[0]
    preferences = {}
    
    column_mapping = {
        'food/café': 'food',
        'transportation': 'transportation', 
        'accommodation': 'accommodation',
        'activities': 'activities',
        'shopping': 'shopping',
        'nightlife': 'nightlife',
        'budget': 'budget'
    }
    
    for csv_col, pref_key in column_mapping.items():
        cell_value = row[csv_col] 
        if pd.notna(cell_value) and cell_value.strip():
            options = [opt.strip() for opt in cell_value.split(',')]
            preferences[pref_key] = random.choice(options)
        else:
            preferences[pref_key] = None
    
    return preferences

def run_single_trial(trial_id: int, actor_temp: float, total_trials: int, date_str: str):
    log_filepath, logger = setup_logging(trial_id)
    
    logger.info(f"Starting run {trial_id}")
    logger.info(f"Log file: {log_filepath}")

    df = pd.read_csv('data/questions_preferences_extented.csv')

    all_questions = df['Questions'].unique().tolist()
    question_id = dict(zip(df['Questions'], df['Q.ID']))

    phase1 = all_questions[:32]
    random.shuffle(phase1)
    phase2 = all_questions[32:]
    random.shuffle(phase2)
    questions = phase1 + phase2
    logger.info(f"Loaded {len(phase1)} + {len(phase2)} questions")

    actor = ACTOR()
    evaluator = EVALUATOR()
    reflexion = REFLEXION()
    hb_model = HUMAN_BEHAVIOR_MODEL()
    logger.info(f"Models loaded successfully")
    
    long_term_memory = LONG_TERM_MEMORY()

    demographic_df = pd.read_csv('data/haiid_dataset.csv', low_memory=False)
    random_demographic = hb_model.pick_random_demographic(demographic_df)
    logger.info(f"Random demographic selected: {random_demographic}")
  
    for qi in range(len(questions)):
        trial_memory = TRIAL_MEMORY()
        question = questions[qi]
        acceptance_history = []
        
        logger.info(f"\n{'='*80}")
        logger.info(f"QUESTION {qi+1}/{len(questions)}: {question}")
        
        user_preferences = get_user_preferences(question, df)
        logger.info(f"User Preferences: {user_preferences}")
        logger.info(f"{'='*80}")

        for i in range(int(max_reflexion_trials)):
            logger.info(f"\n--- Iteration {i+1}/{max_reflexion_trials} ---")
            
            actor_response, actor_confidence = actor.respond(trial_memory, question)
            logger.info(f"Advice: {actor_response} \nAdvice Confidence {actor_confidence*100:.0f}%")
            
            evaluator_correctness, evaluator_acceptability, overall_agreement, individual_agreements = evaluator.evaluate(
                question, actor_response, agreement_parameters=True, 
                user_preferences=user_preferences, relevant_params=None
            )
            measure_value = 1 if overall_agreement else -1
            
            logger.info(f"Advice correctness: {evaluator_acceptability}")
            logger.info(f"Agreement (Hard): {overall_agreement}")
            logger.info("Agreement criteria (relevant only):")
            if not individual_agreements:
                logger.info("    • No specific criteria were relevant for this question.")
            else:
                for param, score in individual_agreements.items():
                    logger.info(f"    • {param.capitalize()}: {score}")

            food_agreement = individual_agreements.get('food', 0)
            transportation_agreement = individual_agreements.get('transportation', 0)
            accommodation_agreement = individual_agreements.get('accommodation', 0)
            activities_agreement = individual_agreements.get('activities', 0)
            shopping_agreement = individual_agreements.get('shopping', 0)
            nightlife_agreement = individual_agreements.get('nightlife', 0)
            budget_agreement = individual_agreements.get('budget', 0)
            
            if evaluator_acceptability:
                evaluator_g_multiplier = 1
            else:
                evaluator_g_multiplier = -1
                
            g_input = actor_confidence * evaluator_g_multiplier
            g_output = hb_model.get_prediction(confidence=g_input, model_type='g')
            logger.info(f"Uncalibrated confidence: {g_output*100:.0f}%")
            
            a_prediction = hb_model.get_prediction(
                confidence=g_output,
                model_type='a',
                user_features=random_demographic,
                measure=measure_value
            )
            logger.info(f"Advice acceptance probability: {a_prediction*100:.0f}%")
            
            user_acceptance_confirmed = hb_model.confirm_user_acceptability(a_prediction)
            
            reflexion_output = reflexion.generate_reflexion(
                trial_memory=trial_memory,
                latest_actor_answer=actor_response,
                is_correct=evaluator_acceptability, 
                user_acceptability=a_prediction,
                long_term_memory=long_term_memory
            )
            logger.info(f"Reflexion: {reflexion_output}")
        
            loss = loss_function(
                float(evaluator_acceptability), 
                float(overall_agreement if overall_agreement is not None else True), 
                a_prediction
            )
            acceptance_history.append(a_prediction)
            
            trial_memory.add_trial(
                question=question,
                episode=i+1,
                actor_advice=actor_response,
                evaluator_correctness=evaluator_correctness,
                evaluator_acceptability=evaluator_acceptability,
                g_model_output=g_output,
                activation_model_output=a_prediction,
                reflexion_output=reflexion_output,
                actor_confidence=actor_confidence,
                g_input=g_input,
                user_acceptability_confirmed=user_acceptance_confirmed,
                food_agreement=food_agreement,
                transportation_agreement=transportation_agreement,
                accommodation_agreement=accommodation_agreement,
                activities_agreement=activities_agreement,
                shopping_agreement=shopping_agreement,
                nightlife_agreement=nightlife_agreement,
                budget_agreement=budget_agreement,
                overall_agreement=overall_agreement,
                question_id=question_id[question],
                loss=loss
            )
            
            if stopping_rule(loss, acceptance_history):
                logger.info(f"STOPPING RULE: Loss {loss:.4f} meets criterion. Stopping after {i+1} iterations.")
                break
            else:
                logger.info(f"CONTINUE: Evaluator acceptable: {evaluator_acceptability}, User acceptable: {user_acceptance_confirmed}, Loss: {loss:.4f}")
           
        logger.info(f"\nQuestion {qi+1} completed. Total iterations: {len(trial_memory.trials)}")
        long_term_memory.store_trial_memory(trial_memory)
        
    logger.info(f"Run {trial_id} completed. Total questions: {len(questions)}")

    json_filename = f"temp_{actor_temp}_runs_{total_trials}_date_{date_str}.json"
    json_filepath = os.path.join("logs", json_filename)
    if total_trials > 1:
        if os.path.exists(json_filepath):
            with open(json_filepath, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
            
        current_trial_data = {
            f"user_{trial_id}": long_term_memory.to_dict()

        }
        existing_data.append(current_trial_data)
        
        with open(json_filepath, 'w') as f:
            json.dump(existing_data, f, indent=4)
    else:
        with open(json_filepath, 'w') as f:
            json.dump(long_term_memory.to_dict(), f, indent=4)

    logger.info(f"Run {trial_id} data saved to: {json_filepath}")
    return json_filepath

def main(n_trials: int = 1):
    start_time = time.time()
    print(f"Running {n_trials} independent runs")
    
    actor_temp = float(os.getenv("ACTOR_TEMP", 1.5))
    timestamp = datetime.now().strftime("%Y%m%d")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_trials) as executor:
        futures = [executor.submit(run_single_trial, i+1, actor_temp, n_trials, timestamp) for i in range(n_trials)]
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
    
    elapsed = time.time() - start_time
    print(f"All {n_trials} runs completed")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print("Results saved to:")
    for result in results:
        print(f"  {result}")

if __name__ == "__main__":
    import sys
    n_trials = 1
    
    if len(sys.argv) > 1:
        try:
            n_trials = int(sys.argv[1])
        except ValueError:
            print("Invalid trial count. Using default of 1.")
            
    print(f"Running {n_trials} independent runs")
    
    with keep.running():
        main(n_trials)
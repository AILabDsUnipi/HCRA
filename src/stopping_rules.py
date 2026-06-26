import math

def loss_function(correctness: float, overall_agreement: float, acceptance_prob: float) -> float:
    if [correctness, overall_agreement] == [1, 1]:
        return -math.log(acceptance_prob)
    else:
        return -math.log(1 - acceptance_prob)
   
def stopping_rule(loss, acceptance_history, e_const: float = 0.01) -> bool:
    print(acceptance_history)
    
    if len(acceptance_history) == 0:
        return False
    
    past_rej_prob = 1.0
    for acceptance_prob in acceptance_history[:-1]:
        past_rej_prob *= (1 - acceptance_prob)

    return loss < e_const/(acceptance_history[-1] * past_rej_prob)
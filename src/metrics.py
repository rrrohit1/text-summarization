from rouge import Rouge
from bert_score import BERTScorer
import pandas as pd

def calculate_metrics(reference_1, reference_2):
    """
    Calculates the ROUGE and BERT scores for the given reference texts.

    Args:
        reference_1 (str): The first reference text.
        reference_2 (str): The second reference text.

    Returns:
        pandas.DataFrame: A DataFrame containing the calculated ROUGE and BERT scores.

    """

    # Initialize the Rouge object
    rouge = Rouge()

    # Calculate the ROUGE scores
    scores = rouge.get_scores(reference_1, reference_2, avg=True)

    rouge_scores = pd.DataFrame(scores)    

    # Calculate the BERTScore
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score(reference_1, reference_2)

    bert_scores = pd.DataFrame({'p': P.mean().item(), 'r': R.mean().item(), 'f': F1.mean().item()}, index=['BERTScore']).T

    return pd.concat([rouge_scores.round(3), bert_scores.round(3)], axis=1)
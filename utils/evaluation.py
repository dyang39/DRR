from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import re
import string
import ast
import collections
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from utils.text_processing import *
from collections import Counter

# Function to normalize text: Lowercase, remove articles, punctuation, and extra spaces
def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(str(s)))))

# Function to compute exact match for multiple correct answers
def exact_match_multiple(prediction, ground_truths):
    """Returns 1 if the prediction matches any ground truth answer exactly, otherwise 0."""
    normalized_prediction = normalize_answer(prediction)
    
    for truth in ground_truths:
        normalized_truth = normalize_answer(truth)
        if normalized_prediction == normalized_truth:
            return 1
    return 0

# Function to get F1 score of a results file (assumes 'df' input is already only the final predictions)
def get_f1_score(df):
    copy = df.copy()
    
    # Initialize variables to accumulate F1 scores
    total_f1 = 0
    total_predictions = len(copy)
    
    # Iterate through the last entries and compute F1 score for each
    for index, row in copy.iterrows():
        # Assuming 'Prediction' contains the model's predicted answer and 'Correct Answer' is a list of gold answers
        prediction = row['Reasoner Answer']
        gold_answers = safe_literal_eval(row['Gold Answer'])  # This should be a list of answers
        
        # Compute F1 score for the current prediction
        f1 = compute_f1_multiple(prediction, gold_answers)
        total_f1 += f1
    
    # Calculate average F1 score
    average_f1 = total_f1 / total_predictions if total_predictions > 0 else 0
    return average_f1

# Function to get accuracy from zero shot or DRR
def get_accuracy(df):
    last_entries = df.groupby('ID').tail(1)
    system_accuracy = accuracy_score(last_entries['Correct Answer'], [1] * len(last_entries))
    correct_count = (last_entries['Correct Answer'] == 1).sum()
    print(f"Number of correct answers (Correct Answer = 1): {correct_count}")

    print(f"Total: {len(last_entries)}")
    print(f"System Accuracy:{system_accuracy:.4f}")
    print(f"F1:{get_f1_score(last_entries):.4f}")
    return system_accuracy

# Function to get zero shot or DRR formula score
def get_formula_score(df, zero_shot=False):
    last_entries = df.groupby('ID').tail(1)
    copy = last_entries.copy()
    copy["Correct Answer"] = copy["Correct Answer"].replace(0, -1)
    # Iterate over the rows
    if zero_shot:
        for index, entry in copy.iterrows():
            if normalize_answer(entry['Reasoner Answer']) == "no answer" or normalize_answer(entry['Reasoner Answer']) == "none" or normalize_answer(entry['Reasoner Answer']) == "-1" or normalize_answer(entry['Reasoner Answer']) == "unsure" or normalize_answer(entry['Reasoner Answer']) == "none of the above" or normalize_answer(entry['Reasoner Answer']) == "unanswerable":
                copy.at[index, 'Correct Answer'] = 0
    else:
        assert 'Gate Output' in copy.columns, "Gate Output must be a column"
        for index, entry in copy.iterrows():
            # Check if 'Gate Output' is False and update 'Correct Answer'
            if entry['Gate Output'] == False:
                copy.at[index, 'Correct Answer'] = 0
            
    # Print the number of last entries
    num_entries = len(last_entries)
    print(f"Number of eval points: {num_entries}")

    # Print the value counts for the 'Correct Answer' column
    correct_answer_counts = copy['Correct Answer'].value_counts()
    print("\nCorrect Answer value counts:")
    print(correct_answer_counts)

    # Calculate and print the sum of the 'Correct Answer' column
    formula_score = copy['Correct Answer'].sum()
    print(f"\nTotal formula score: {formula_score}")

    # Calculate and print the score divided by the number of entries
    average_score = formula_score / num_entries
    print(f"Formula percentage (average): {average_score:.4f}")
    return average_score

# Gives more information about DM performance during DRR
def custom_evaluation(df, name = ''):
    last_entries = df.groupby('ID').tail(1)
    system_accuracy = accuracy_score(last_entries['Correct Answer'], [1] * len(last_entries))
    f1 = get_f1_score(last_entries)
    formula_score = get_formula_score(last_entries)

    # Gate Accuracy, Precision, Recall, F1, False Positive Rate, True Negative Rate
    gate_accuracy = accuracy_score(df['Correct Answer'], df['Gate Output'])
    gate_precision = precision_score(df['Correct Answer'], df['Gate Output'])
    gate_recall = recall_score(df['Correct Answer'], df['Gate Output'])
    gate_f1 = f1_score(df['Correct Answer'], df['Gate Output'])
    
    tn, fp, fn, tp = confusion_matrix(df['Correct Answer'], df['Gate Output']).ravel()
    gate_false_positive_rate = fp / (fp + tn)
    gate_true_negative_rate = tn / (tn + fp)

    results = {
        'Num Data Points': len(df),
        '------------------------':'',
        'System Accuracy': system_accuracy,
        'F1': f1,
        'Formula Score': formula_score,
        '-------------------------':'',
        'Gate Accuracy': gate_accuracy,
        'Gate Precision': gate_precision,
        'Gate Recall': gate_recall,
        'Gate F1 Score': gate_f1,
        'Gate False Positive Rate': gate_false_positive_rate,
        'Gate True Negative Rate': gate_true_negative_rate,
        '--------------------------':'',
    }
    
    # Print Results
    print()
    print()
    print(f"===== Custom Evaluation Results for {name} =====")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        elif isinstance(value,int):
            print(f"{metric}: {value}")
        elif value is None:
            print(f"{metric}: No valid entries")
        else:
            print(f"{metric}")

    print("===================================================")
    return results

# Gets the accuracy at each turn in predictions, including zero shot accuracy during DRR
def evaluate_per_turn(df,prompt_setting="exploration"):
    
    # Print the original distribution
    print("Original Distribution:")
    print(len(df))
    print(df['Gate Output'].value_counts())  # Total count of Gate Verdict 0 and 1
    print(df['Correct Answer'].value_counts()) 

    # Group the data by 'ID'
    grouped = df.groupby('ID')

    # Group by 'ID' and get the size of each group
    group_sizes = grouped.size()

    # Print the number of groups of each size
    print("\nGroup Size Distribution:")
    print(group_sizes.value_counts().sort_index())  # Count of groups by size

    # Group by 'Turn' and count the occurrences of each 'Gate Output' value
    turn_counts = df.groupby('Turn')['Gate Output'].value_counts().unstack(fill_value=0)

    # Print the results
    print("Gate Output counts per Turn:")
    print(turn_counts)

    # Group by 'Turn' and count the occurrences of each 'Gate Output' value
    turn_counts = df.groupby('Turn')['Correct Answer'].value_counts().unstack(fill_value=0)

    # Print the results
    print("Correct Answer counts per Turn:")
    print(turn_counts)

    # Calculate zero-shot accuracy for Turn = 0
    if prompt_setting == "gradual":
        turn_0 = df[df['Turn'] == 1]
    else:
        turn_0 = df[df['Turn'] == 0]
    total_turn_0 = len(turn_0)
    
    if total_turn_0 > 0:
        correct_turn_0 = turn_0['Correct Answer'].sum()  # Count of 1s in 'Gate Output'
        zero_shot_accuracy = correct_turn_0 / total_turn_0  # Calculate accuracy
        print(f"Zero Shot Accuracy:{zero_shot_accuracy:.4f}")
        print(f"Zero Shot F1:{get_f1_score(turn_0):.4f}")
    else:
        print("No data to calculate Zero Shot Accuracy.")


# Measures LLM's redundancy in predictions each turn
def measure_redundancy(df, overlap_threshold=20):
    total_groups = df["ID"].nunique()
    total_answers = 0
    total_rationales = 0
    redundant_answers = 0
    redundant_rationales = 0
    
    answer_redundancy_counts = Counter()  # Tracks how many groups have X repeated answers
    rationale_redundancy_counts = Counter()  # Tracks how many groups have X overlapping rationales
    
    # Group by "ID"
    grouped = df.groupby("ID")
    
    for group_id, group in grouped:
        #rint(f"\nProcessing ID: {group_id}")
        
        # 1) Count how many values in the group have the same "Reasoner Answer"
        answer_counts = group["Reasoner Answer"].value_counts()
        group_size = len(group)
        total_answers += group_size
        group_redundant_answers = sum(count for count in answer_counts if count > 1)
        redundant_answers += group_redundant_answers
        
        # Update the redundancy count for the group
        answer_redundancy_counts[group_redundant_answers] += 1
        
        #print(f"Group {group_id} - Redundant Answers: {group_redundant_answers}")
        
        # 2) Count how many "Reasoner Rationale" have overlapping words with other "Reasoner Rationale"
        rationale_texts = group["Reasoner Rationale"].dropna().tolist()
        total_rationales += len(rationale_texts)
        
        group_redundant_rationales = 0  # Count redundant rationales
        if len(rationale_texts) > 1:
            # Use CountVectorizer to get the word count for each rationale
            vectorizer = CountVectorizer().fit_transform(rationale_texts)
            vectors = vectorizer.toarray()
            
            # Compare each pair of rationales for word overlap
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    # Count overlapping words between two rationales
                    word_overlap = (vectors[i] & vectors[j]).sum()
                    
                    # If overlap exceeds the threshold, count it as redundant
                    if word_overlap >= overlap_threshold:
                        group_redundant_rationales += 1
                        break  # Only count the rationale once
            
            redundant_rationales += group_redundant_rationales
            rationale_redundancy_counts[group_redundant_rationales] += 1  # Update rationale redundancy count

        
        # Count this group as having 0 overlaps if no redundant rationales were found
        if group_redundant_rationales == 0:
            rationale_redundancy_counts[0] += 1

    # Calculate overall percentages
    answer_redundancy_percentage = (redundant_answers / total_answers) * 100 if total_answers > 0 else 0
    rationale_redundancy_percentage = (redundant_rationales / total_rationales) * 100 if total_rationales > 0 else 0
    
    print("\nOverall Redundancy:")
    print(f"Answer Redundancy: {answer_redundancy_percentage:.2f}%")
    print(f"Rationale Redundancy ({overlap_threshold} overlapping): {rationale_redundancy_percentage:.2f}%")
    
    def print_redundancy_counts(counts, label):
        print(f"\n{label} (How many groups have X repetitions/overlaps):")
        for key, value in sorted(counts.items(), reverse=True):
            print(f"  {value} group(s) with {key} repeated answer values/overlapping rationale(s)")

    print_redundancy_counts(answer_redundancy_counts, "Answer Redundancy Counts")
    print_redundancy_counts(rationale_redundancy_counts, f"Rationale Redundancy Counts ({overlap_threshold} overlapping)")
    
    return {
        "Overall Answer Redundancy (%)": answer_redundancy_percentage,
        "Overall Rationale Redundancy (%)": rationale_redundancy_percentage,
        "Answer Redundancy Counts": answer_redundancy_counts,
        "Rationale Redundancy Counts": rationale_redundancy_counts
    }

# Get info/stats about generated data
def per_turn_generation(df, prompt_setting="exploration"):
    # Group the data by 'ID'
    grouped = df.groupby('ID')

    # Group by 'ID' and get the size of each group
    group_sizes = grouped.size()

    # Print the number of groups of each size
    #print("\nGroup Size Distribution:")
    #print(group_sizes.value_counts().sort_index())  # Count of groups by size

    # Group by 'Turn' and count the occurrences of each 'Gate Output' value
    turn_counts = df.groupby('Turn')['Verdict'].value_counts().unstack(fill_value=0)

    # Print the results
    print("Verdict Distribution per Turn")
    print(turn_counts)

    # Calculate zero-shot accuracy for Turn = 0
    if prompt_setting == "gradual":
        turn_0 = df[df['Turn'] == 1]
    else:
        turn_0 = df[df['Turn'] == 0]
    total_turn_0 = len(turn_0)
    
    if total_turn_0 > 0:
        correct_turn_0 = turn_0['Verdict'].sum()  # Count of 1s in 'Gate Output'
        zero_shot_accuracy = correct_turn_0 / total_turn_0  # Calculate accuracy
        zero_shot_f1 =get_f1_score(turn_0)
        print("Zero Shot Accuracy: {:.2f}%".format(zero_shot_accuracy * 100))
        print(f"Zero Shot F1:{zero_shot_f1:.4f}")
    else:
        print("No data to calculate Zero Shot Accuracy.")
    return turn_counts,zero_shot_accuracy,zero_shot_f1

# Get accuracy from generated data
def get_accuracy_generation(df):
    last_entries = df.groupby('ID').tail(1)
    system_accuracy = accuracy_score(last_entries['Verdict'], [1] * len(last_entries))
    
    print(f"System Accuracy:{system_accuracy:.4f}")
    print(f"F1:{get_f1_score(last_entries):.4f}")
    return system_accuracy


# F1 helper functions
def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()
  
# Compute F1 score for a prediction against multiple correct answers
def compute_f1_multiple(a_pred, gold_answers):
    pred_toks = get_tokens(a_pred)
    gold_toks_list = [get_tokens(ans) for ans in gold_answers]
    
    # Find the best F1 score across all gold answers
    best_f1 = 0
    for gold_toks in gold_toks_list:
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            best_f1 = max(best_f1, int(gold_toks == pred_toks))
        elif num_same != 0:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            best_f1 = max(best_f1, f1)
    
    return best_f1

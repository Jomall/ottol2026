import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import random

# Copy the generate_sequence function here for testing
def generate_sequence(df):
    # Prepare data for ML
    # For supervised: Predict each position based on previous (simple chain)
    X = []
    y = []
    X_bb_letter = []
    y_bb = []
    y_letter = []
    X_sup = []
    y_sup = []
    X_sup_bb_letter = []
    y_sup_bb = []
    y_sup_letter = []
    for i in range(len(df) - 1):
        seq = df.iloc[i][['Num1', 'Num2', 'Num3', 'Num4', 'Num5']].values.tolist()
        next_seq = df.iloc[i+1]
        for pos in range(5):
            X.append((seq[:pos] + [0] * (5 - pos)) if pos > 0 else [0] * 5)  # Features: previous positions, padded to 5
            y.append(next_seq.iloc[pos+2])  # Target: next position's number (Num1 is column 2)
        # For BB and Letter, use the full previous sequence as features
        X_bb_letter.append(seq)
        y_bb.append(next_seq['BB'])
        y_letter.append(next_seq['Letter'])

        # For SUP6, if available
        if not pd.isna(next_seq['Sup1']) and next_seq['Sup1'] != 0:  # Assuming 0 means not parsed
            sup_seq = df.iloc[i][['Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6']].values.tolist()
            for pos in range(6):
                X_sup.append((sup_seq[:pos] + [0] * (6 - pos)) if pos > 0 else [0] * 6)  # Features: previous SUP6 positions, padded to 6
                y_sup.append(next_seq[f'Sup{pos+1}'])
            X_sup_bb_letter.append(seq)
            y_sup_bb.append(next_seq['SupBB'])
            y_sup_letter.append(next_seq['SupLetter'])

    X = np.array(X)
    y = np.array(y)
    X_bb_letter = np.array(X_bb_letter)
    y_bb = np.array(y_bb)
    y_letter = np.array(y_letter)
    if len(X_sup) > 0:
        X_sup = np.array(X_sup)
        y_sup = np.array(y_sup)
        X_sup_bb_letter = np.array(X_sup_bb_letter)
        y_sup_bb = np.array(y_sup_bb)
        y_sup_letter = np.array(y_sup_letter)

    # Train a simple model (Random Forest for regression on numbers)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Train model for BB
    model_bb = RandomForestRegressor(n_estimators=100, random_state=42)
    model_bb.fit(X_bb_letter, y_bb)

    # Train model for Letter
    model_letter = RandomForestClassifier(n_estimators=100, random_state=42)
    model_letter.fit(X_bb_letter, y_letter)

    # Train SUP6 models if data available
    if len(X_sup) > 0:
        model_sup = RandomForestRegressor(n_estimators=100, random_state=42)
        model_sup.fit(X_sup, y_sup)
        model_sup_bb = RandomForestRegressor(n_estimators=100, random_state=42)
        model_sup_bb.fit(X_sup_bb_letter, y_sup_bb)
        model_sup_letter = RandomForestClassifier(n_estimators=100, random_state=42)
        model_sup_letter.fit(X_sup_bb_letter, y_sup_letter)

    # Unsupervised: Cluster sequences to find patterns
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB']].values)

    # Generate new sequence: Start with a random seed, predict step-by-step, bias toward high-freq numbers, ensure no duplicates
    new_seq = []
    used = set()
    prev = [0] * 5  # Start with padded features
    for pos in range(5):
        attempts = 0
        while attempts < 10:  # Limit attempts to prevent infinite loop
            pred = model.predict(np.array([prev]))[0]
            # Bias: Adjust toward most frequent in that position
            freq = df.iloc[:, pos + 2].value_counts()  # Num1 is column 2
            if freq.empty:
                top_num = 1  # Default if no data
            else:
                top_num = freq.idxmax()
            # Handle NaN prediction
            if np.isnan(pred):
                pred = top_num  # Use top_num if prediction is NaN
            # Blend prediction with frequency (e.g., 70% pred, 30% top freq)
            blended = int(0.7 * pred + 0.3 * top_num)
            blended = max(1, min(36, blended))  # Clamp to 1-36
            if blended not in used:
                new_seq.append(blended)
                used.add(blended)
                prev = new_seq + [0] * (5 - len(new_seq))  # Update prev with current sequence padded
                break
            else:
                # Try next most frequent not used
                freq_sorted = freq.sort_values(ascending=False)
                for num in freq_sorted.index:
                    if num not in used:
                        blended = int(num)
                        new_seq.append(blended)
                        used.add(blended)
                        prev = new_seq + [0] * (5 - len(new_seq))
                        break
                else:
                    # If all frequent used, pick random available
                    available = [n for n in range(1, 37) if n not in used]
                    if available:
                        blended = random.choice(available)
                        new_seq.append(blended)
                        used.add(blended)
                        prev = new_seq + [0] * (5 - len(new_seq))
                    break
            attempts += 1
        if attempts >= 10:
            # Fallback: random unique if still not found
            available = [n for n in range(1, 37) if n not in used]
            if available:
                blended = random.choice(available)
                new_seq.append(blended)
                used.add(blended)
                prev = new_seq + [0] * (5 - len(new_seq))

    # Predict BB and Letter based on the generated sequence
    pred_bb = model_bb.predict(np.array([new_seq]))[0]
    pred_bb = int(max(1, min(36, pred_bb)))  # Clamp BB to reasonable range, assuming 1-36
    pred_letter = model_letter.predict(np.array([new_seq]))[0]

    # Generate SUP6 if model available, ensure no duplicates
    sup6 = {}
    if len(X_sup) > 0:
        new_sup_seq = []
        used_sup = set()
        prev_sup = [0] * 6
        for pos in range(6):
            attempts = 0
            while attempts < 10:  # Limit attempts to prevent infinite loop
                pred_sup = float(model_sup.predict(np.array([prev_sup]))[0])
                # Bias toward freq
                freq_sup = df.iloc[:, 9 + pos].value_counts()  # Sup1 to Sup6 are columns 9 to 14
                if freq_sup.empty:
                    top_num_sup = 1.0
                else:
                    top_val = freq_sup.idxmax()
                    try:
                        top_num_sup = float(top_val)
                    except ValueError:
                        top_num_sup = 1.0  # Default if not numeric
                if np.isnan(pred_sup):
                    pred_sup = top_num_sup
                blended_sup = int(0.7 * pred_sup + 0.3 * top_num_sup)
                blended_sup = max(1, min(36, blended_sup))
                if blended_sup not in used_sup:
                    new_sup_seq.append(blended_sup)
                    used_sup.add(blended_sup)
                    prev_sup = new_sup_seq + [0] * (6 - len(new_sup_seq))
                    break
                else:
                    # Try next most frequent not used
                    freq_sorted_sup = freq_sup.sort_values(ascending=False)
                    for num in freq_sorted_sup.index:
                        if num not in used_sup:
                            blended_sup = int(num)
                            new_sup_seq.append(blended_sup)
                            used_sup.add(blended_sup)
                            prev_sup = new_sup_seq + [0] * (6 - len(new_sup_seq))
                            break
                    else:
                        # If all frequent used, pick random available
                        available_sup = [n for n in range(1, 37) if n not in used_sup]
                        if available_sup:
                            blended_sup = random.choice(available_sup)
                            new_sup_seq.append(blended_sup)
                            used_sup.add(blended_sup)
                            prev_sup = new_sup_seq + [0] * (6 - len(new_sup_seq))
                        break
                attempts += 1
            if attempts >= 10:
                # Fallback: random unique if still not found
                available_sup = [n for n in range(1, 37) if n not in used_sup]
                if available_sup:
                    blended_sup = random.choice(available_sup)
                    new_sup_seq.append(blended_sup)
                    used_sup.add(blended_sup)
                    prev_sup = new_sup_seq + [0] * (6 - len(new_sup_seq))
        pred_sup_bb = model_sup_bb.predict(np.array([new_seq]))[0]
        pred_sup_bb = int(max(1, min(36, pred_sup_bb)))
        pred_sup_letter = model_sup_letter.predict(np.array([new_seq]))[0]
        sup6 = {'numbers': new_sup_seq, 'bb': pred_sup_bb, 'letter': pred_sup_letter}

    return {'numbers': new_seq, 'bb': pred_bb, 'letter': pred_letter, 'sup6': sup6}, model, kmeans

# Create sample data
data = {
    'NLA': [
        '1 2 3 4 5 BB: 6 Letter: A',
        '2 3 4 5 6 BB: 7 Letter: B',
        '3 4 5 6 7 BB: 8 Letter: C',
        '4 5 6 7 8 BB: 9 Letter: D',
        '5 6 7 8 9 BB: 10 Letter: E',
        '6 7 8 9 10 BB: 11 Letter: F',
        '7 8 9 10 11 BB: 12 Letter: G',
        '8 9 10 11 12 BB: 13 Letter: H',
        '9 10 11 12 13 BB: 14 Letter: I',
        '10 11 12 13 14 BB: 15 Letter: J'
    ],
    'SUP6': [
        '11 12 13 14 15 16 BB: 17 Letter: K',
        '12 13 14 15 16 17 BB: 18 Letter: L',
        '13 14 15 16 17 18 BB: 19 Letter: M',
        '14 15 16 17 18 19 BB: 20 Letter: N',
        '15 16 17 18 19 20 BB: 21 Letter: O',
        '16 17 18 19 20 21 BB: 22 Letter: P',
        '17 18 19 20 21 22 BB: 23 Letter: Q',
        '18 19 20 21 22 23 BB: 24 Letter: R',
        '19 20 21 22 23 24 BB: 25 Letter: S',
        '20 21 22 23 24 25 BB: 26 Letter: T'
    ]
}
df = pd.DataFrame(data)

# Parse as in app
parsed_data = []
for _, row in df.iterrows():
    nla = row['NLA']
    import re
    numbers = re.findall(r'\d+', nla)
    nums = [int(n) for n in numbers[:5]]
    bb_match = re.search(r'BB[:\-]\s*(\d+)', nla)
    letter_match = re.search(r'Letter:\s*([A-Z])', nla)
    bb = int(bb_match.group(1)) if bb_match else 1
    letter = letter_match.group(1) if letter_match else 'A'

    sup_parsed = [0] * 8
    if 'SUP6' in df.columns and not pd.isna(row['SUP6']) and isinstance(row['SUP6'], str):
        sup6 = row['SUP6']
        numbers_sup = re.findall(r'\d+', sup6)
        if len(numbers_sup) >= 6:
            nums_sup = [int(n) for n in numbers_sup[:6]]
            bb_match_sup = re.search(r'BB[:\-]\s*(\d+)', sup6)
            letter_match_sup = re.search(r'Letter:\s*([A-Z])', sup6)
            bb_sup = int(bb_match_sup.group(1)) if bb_match_sup else 1
            letter_sup = letter_match_sup.group(1) if letter_match_sup else 'A'
            sup_parsed = nums_sup + [bb_sup, letter_sup]

    parsed_data.append([None, None] + nums + [bb, letter] + sup_parsed)
df_parsed = pd.DataFrame(parsed_data, columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupBB', 'SupLetter'])

# Generate sequence
sequence, _, _ = generate_sequence(df_parsed)

# Test for duplicates
main_seq = sequence['numbers']
sup_seq = sequence.get('sup6', {}).get('numbers', [])

print("Generated Main Sequence:", main_seq)
print("Duplicates in Main Sequence:", len(main_seq) != len(set(main_seq)))
print("All in 1-36:", all(1 <= n <= 36 for n in main_seq))

if sup_seq:
    print("Generated SUP6 Sequence:", sup_seq)
    print("Duplicates in SUP6 Sequence:", len(sup_seq) != len(set(sup_seq)))
    print("All in 1-36:", all(1 <= n <= 36 for n in sup_seq))
else:
    print("No SUP6 generated")

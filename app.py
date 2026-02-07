
from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import os
import random
import math
import itertools
from collections import Counter
from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mutual_info_score
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

app = Flask(__name__)
# For Vercel deployment, use temporary directory
import tempfile
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables to store data and model
global_df = None
global_model = None
global_kmeans = None
global_csv_path = None
global_csv_filename = None
global_current_sequence = None
global_unsaved_changes = False

# Persistent save path
SAVE_PATH = r'C:\Users\jomal\OneDrive\Documents\reconstructed_lotto.csv'

def load_saved_data():
    save_path = r'C:\Users\jomal\OneDrive\Documents\reconstructed_lotto.csv'
    if os.path.exists(save_path):
        return load_existing_df(save_path)
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    global global_df, global_model, global_kmeans, global_csv_path, global_csv_filename
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            file.save(filepath)

            # Load and validate data
            df = pd.read_csv(filepath)
            if 'NLA' not in df.columns:
                return "Error: CSV must have a column named 'NLA'."
            if len(df) < 1:
                return "Error: CSV must have at least 1 row."

            # Parse NLA column to extract 5 numbers, BB, and Letter per row
            parsed_data = []
            for _, row in df.iterrows():
                nla = row['NLA']
                if pd.isna(nla) or not isinstance(nla, str):
                    continue
                # Extract numbers from NLA string
                import re
                numbers = re.findall(r'\d+', nla)
                if len(numbers) < 5:
                    return f"Error: Row '{nla}' does not contain at least 5 numbers."
                nums = [int(n) for n in numbers[:5]]
                if not all(0 <= n <= 36 for n in nums):
                    return f"Error: Numbers in '{nla}' must be between 0 and 36."
                # Extract BB and Letter, add defaults if missing
                bb_match = re.search(r'BB[:\-]\s*(\d+)', nla)
                letter_match = re.search(r'Letter:\s*([A-Z])', nla)
                bb = int(bb_match.group(1)) if bb_match else 1
                letter = letter_match.group(1) if letter_match else 'A'

                # Parse SUP6 if present
                sup_parsed = ['00'] * 6 + ['A']  # 6 nums + letter
                if 'SUP6' in df.columns and not pd.isna(row['SUP6']) and isinstance(row['SUP6'], str):
                    sup6 = row['SUP6']
                    numbers_sup = re.findall(r'\d+', sup6)
                    if len(numbers_sup) >= 6:
                        nums_sup = [n.zfill(2) for n in numbers_sup[:6]]
                        if all(1 <= int(n) <= 36 for n in nums_sup):
                            letter_match_sup = re.search(r'Letter:\s*([A-Z])', sup6)
                            letter_sup = letter_match_sup.group(1) if letter_match_sup else 'A'
                            # Check for letter at the end if not found in regex
                            if letter_match_sup is None:
                                parts = sup6.split()
                                for part in reversed(parts):
                                    if part.isalpha() and len(part) == 1:
                                        letter_sup = part.upper()
                                        break
                            sup_parsed = nums_sup + [letter_sup]

                # Include Draw and Draw# if available
                draw_val = row.get('Draw ', None)
                draw_num_val = row.get('Draw#', None)
                parsed_data.append([draw_val, draw_num_val] + nums + [bb, letter] + sup_parsed)
            new_df = pd.DataFrame(parsed_data, columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupLetter'])

            # Load existing data from SAVE_PATH and append new data without duplicates
            global_csv_path = SAVE_PATH  # Always save to persistent path
            global_csv_filename = file.filename  # Save the filename for display
            existing_df = load_existing_df(SAVE_PATH)
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Drop duplicates based on Draw and Draw# to avoid duplicate entries
                global_df = combined_df.drop_duplicates(subset=['Draw', 'Draw#'], keep='first').reset_index(drop=True)
            else:
                global_df = new_df.copy()

            # Store globally for recalibration
            global_model = None
            global_kmeans = None

            # Analysis
            analysis = analyze_patterns(global_df)

            # ML and Generation
            generated_sequence, model, kmeans = generate_sequence(global_df)
            global_model = model
            global_kmeans = kmeans
            global_current_sequence = generated_sequence

            # Save updated data to SAVE_PATH
            try:
                convert_df_to_sample_format(global_df).to_csv(SAVE_PATH, index=False)
            except Exception as e:
                print(f"Error saving to {SAVE_PATH}: {e}")

            # Save prediction to history
            save_prediction_to_history(generated_sequence)

            # Calculate last NLA and SUP6, and totals
            last_nla = None
            last_sup6 = None
            total_nla = 0
            total_sup6 = 0

            if not global_df.empty:
                # Last NLA: row with max Draw
                nla_rows = global_df[global_df['Draw'].notna()]
                if not nla_rows.empty:
                    last_nla_row = nla_rows.loc[nla_rows['Draw'].idxmax()]
                    last_nla = f"Draw {int(last_nla_row['Draw'])}: {int(last_nla_row['Num1'])} {int(last_nla_row['Num2'])} {int(last_nla_row['Num3'])} {int(last_nla_row['Num4'])} {int(last_nla_row['Num5'])} BB: {int(last_nla_row['BB'])} Letter: {last_nla_row['Letter']}"
                    total_nla = len(nla_rows)

                # Last SUP6: row with max Draw#
                sup6_rows = global_df[global_df['Draw#'].notna()]
                if not sup6_rows.empty:
                    last_sup6_row = sup6_rows.loc[sup6_rows['Draw#'].idxmax()]
                    last_sup6 = f"Draw# {int(last_sup6_row['Draw#'])}: {last_sup6_row['Sup1']} {last_sup6_row['Sup2']} {last_sup6_row['Sup3']} {last_sup6_row['Sup4']} {last_sup6_row['Sup5']} {last_sup6_row['Sup6']} Letter: {last_sup6_row['SupLetter']}"
                    total_sup6 = len(sup6_rows)

            return render_template('results.html', analysis=analysis, sequence=generated_sequence, csv_filename=global_csv_filename, last_nla=last_nla, last_sup6=last_sup6, total_nla=total_nla, total_sup6=total_sup6)
    return render_template('index.html')

def calculate_probability(n=5, k=36):
    """Calculate total combinations and odds for lotto game."""
    total_combinations = math.comb(k, n)
    odds = total_combinations
    return {'total_combinations': total_combinations, 'odds': odds}

def monte_carlo_simulation(df, num_simulations=1000):
    """Run Monte Carlo simulations based on historical frequencies."""
    all_nums = []
    for _, row in df.iterrows():
        all_nums.extend([row['Num1'], row['Num2'], row['Num3'], row['Num4'], row['Num5']])
    freq = Counter(all_nums)
    total_draws = len(df)
    probabilities = {num: count / (total_draws * 5) for num, count in freq.items()}

    # Simulate draws
    simulated_numbers = []
    for _ in range(num_simulations):
        draw = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=5)
        simulated_numbers.extend(draw)

    # Return average of simulated numbers
    sim_freq = Counter(simulated_numbers)
    avg_nums = [num for num, _ in sim_freq.most_common(5)]
    return avg_nums

def avoid_popular_patterns(sequence):
    """Avoid common patterns like dates or arithmetic sequences."""
    numbers = sequence['numbers']
    # Check for date patterns (1-31)
    if all(1 <= n <= 31 for n in numbers):
        # Replace with less common numbers
        cold_nums = [32, 33, 34, 35, 36]
        sequence['numbers'] = cold_nums[:5]

    # Check for arithmetic sequences (e.g., 1,2,3,4,5)
    diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
    if len(set(diffs)) == 1:  # All differences equal
        # Replace with random non-arithmetic
        sequence['numbers'] = random.sample(range(1, 37), 5)

    return sequence

def wheel_system(numbers, k=5):
    """Generate multiple combinations using wheel system."""
    if len(numbers) < k:
        return [numbers]
    # Simple wheel: all combinations of k from numbers
    return list(itertools.combinations(numbers, k))

def analyze_patterns(df):
    results = {}

    # Fundamental Probability
    prob = calculate_probability()
    results['Total Combinations'] = prob['total_combinations']
    results['Odds of Winning'] = f"1 in {prob['odds']}"

    # Positional frequencies
    for pos in range(5):
        results[f'Position {pos+1} Frequencies'] = df.iloc[:, pos].value_counts().to_dict()

    # Hot and Cold numbers
    all_nums = []
    for _, row in df.iterrows():
        all_nums.extend([row['Num1'], row['Num2'], row['Num3'], row['Num4'], row['Num5']])
    freq = Counter(all_nums)
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    results['Hot Numbers'] = [num for num, count in sorted_freq[:10]]
    results['Cold Numbers'] = [num for num, count in sorted_freq[-10:]]

    # Numbers never called
    all_numbers = set(range(1, 37))
    used_numbers = set(df.values.flatten())
    results['Never Called'] = list(all_numbers - used_numbers)

    # Numbers always called (in all sequences)
    always_called = set(range(1, 37))
    for _, row in df.iterrows():
        always_called &= set([row['Num1'], row['Num2'], row['Num3'], row['Num4'], row['Num5']])
    results['Always Called'] = list(always_called) if always_called else "None"

    # Positional repeats
    repeats = {}
    for pos in range(5):
        pos_data = df.iloc[:, pos]
        repeats[f'Position {pos+1}'] = pos_data.value_counts().max()
    results['Max Repeats per Position'] = repeats

    # Delta System: Differences between consecutive numbers
    deltas = []
    for _, row in df.iterrows():
        nums = sorted([row['Num1'], row['Num2'], row['Num3'], row['Num4'], row['Num5']])
        row_deltas = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
        deltas.extend(row_deltas)
    delta_freq = Counter(deltas)
    results['Common Deltas'] = dict(delta_freq.most_common(5))

    # Co-occurrence Analysis: Pairs of numbers that appear together
    pairs = []
    for _, row in df.iterrows():
        nums = [row['Num1'], row['Num2'], row['Num3'], row['Num4'], row['Num5']]
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                pairs.append(tuple(sorted([nums[i], nums[j]])))
    pair_freq = Counter(pairs)
    results['Common Pairs'] = dict(pair_freq.most_common(10))

    return results

def load_historical_predictions():
    """Load and parse historical predictions to incorporate into algorithm."""
    history_file = 'prediction_history.csv'
    if not os.path.exists(history_file):
        return []

    df_hist = pd.read_csv(history_file)
    predictions = []
    for _, row in df_hist.iterrows():
        nla_pred = row['nla_prediction']
        # Parse NLA prediction: e.g., "5-10-20-28-32 BB: 11 Letter: A"
        import re
        nla_match = re.match(r'(\d+)-(\d+)-(\d+)-(\d+)-(\d+) BB: (\d+) Letter: ([A-Z])', nla_pred)
        if nla_match:
            nums = [int(nla_match.group(i)) for i in range(1, 6)]
            bb = int(nla_match.group(6))
            letter = nla_match.group(7)
            predictions.append({'numbers': nums, 'bb': bb, 'letter': letter})
    return predictions

def generate_sequence(df):
    # Drop rows with any NaN to avoid sklearn errors
    df = df.dropna()

    # Load historical predictions
    hist_predictions = load_historical_predictions()

    # If not enough data, return a random sequence
    if len(df) < 2:
        new_seq = random.sample(range(1, 37), 5)
        pred_bb = random.randint(1, 36)
        pred_letter = random.choice(['A', 'B', 'C', 'D', 'E'])
        return {'numbers': new_seq, 'bb': pred_bb, 'letter': pred_letter, 'sup6': None}, None, None

    # Convert numeric columns to regular int/float to avoid nullable types
    numeric_cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if 'Draw' in df.columns:
        df['Draw'] = df['Draw'].astype(int)
    if 'Draw#' in df.columns:
        df['Draw#'] = df['Draw#'].astype(int)

    # Prepare data for ML
    # For supervised: Predict each position based on previous (simple chain)
    X = []
    y = []
    X_bb_letter = []
    y_bb = []
    y_letter = []
    X_sup = []
    y_sup = []
    X_sup_letter = []
    y_sup_letter = []
    sequences = []
    for i in range(len(df) - 1):
        seq = df.iloc[i, 2:7].values.tolist()
        next_seq = df.iloc[i+1]
        sequences.append(seq)
        for pos in range(5):
            X.append((seq[:pos] + [0] * (9 - pos)) if pos > 0 else [0] * 9)  # Features: previous positions, padded to 9
            y.append(next_seq[pos+2])  # Target: next position's number (Num1 is column 2)
        # For BB and Letter, use the full previous sequence as features
        X_bb_letter.append(seq)
        y_bb.append(next_seq['BB'])
        y_letter.append(next_seq['Letter'])

        # For SUP6, if available
        if not pd.isna(next_seq['Sup1']) and next_seq['Sup1'] != '00':  # Assuming '00' means not parsed
            sup_values = df.iloc[i][['Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6']].values.tolist()
            if not any(pd.isna(s) for s in sup_values):
                sup_seq = [int(s) for s in sup_values]
                for pos in range(6):
                    X_sup.append((sup_seq[:pos] + [0] * (6 - pos)) if pos > 0 else [0] * 6)  # Features: previous SUP6 positions, padded to 6
                    y_sup.append(int(next_seq[f'Sup{pos+1}']))
            X_sup_letter.append(seq)
            y_sup_letter.append(next_seq['SupLetter'])

    X = np.array(X)
    y = np.array(y)
    X_bb_letter = np.array(X_bb_letter)
    y_bb = np.array(y_bb)
    y_letter = np.array(y_letter)
    if len(X_sup) > 0:
        X_sup = np.array(X_sup)
        y_sup = np.array(y_sup)
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
        model_sup_letter = RandomForestClassifier(n_estimators=100, random_state=42)
        model_sup_letter.fit(X_sup_letter, y_sup_letter)

    # LSTM for sequence prediction
    lstm_model = None
    if TENSORFLOW_AVAILABLE and len(sequences) > 10:  # Need enough data for LSTM
        try:
            # Prepare sequences for LSTM: sequences of numbers over time
            seq_length = 5  # Look at last 5 draws
            X_lstm = []
            y_lstm = []
            for i in range(len(sequences) - seq_length):
                X_lstm.append(sequences[i:i+seq_length])
                y_lstm.append(sequences[i+seq_length])
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)

            # Build LSTM model
            lstm_model = Sequential()
            lstm_model.add(LSTM(50, activation='relu', input_shape=(seq_length, 5)))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(Dense(5))  # Predict 5 numbers
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
        except Exception as e:
            print(f"LSTM training failed: {e}")
            lstm_model = None

    # Unsupervised: Cluster sequences to find patterns
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB']].values)

    # Incorporate historical predictions for improved accuracy
    hist_freq = {}
    if hist_predictions:
        for pred in hist_predictions:
            for num in pred['numbers']:
                hist_freq[num] = hist_freq.get(num, 0) + 1
        # Sort by frequency
        hist_sorted = sorted(hist_freq.items(), key=lambda x: x[1], reverse=True)

    # Generate new sequence: Start with a random seed, predict step-by-step, bias toward high-freq numbers, ensure no duplicates
    new_seq = []
    used = set()
    prev = [0] * 9  # Start with padded features
    for pos in range(5):
        pred = model.predict(np.array([prev]))[0]
        # Use LSTM if available for better prediction
        # if lstm_model is not None:
        #     lstm_pred = lstm_model.predict(np.array([prev]).reshape(1, 5, 1))[0]
        #     pred = 0.5 * pred + 0.5 * np.mean(lstm_pred)  # Blend RF and LSTM predictions
        # Bias: Adjust toward most frequent in that position
        freq = df.iloc[:, pos + 2].value_counts()  # Num1 is column 2
        if freq.empty:
            top_num = 1  # Default if no data
        else:
            top_num = freq.idxmax()
        # Handle NaN prediction
        if np.isnan(pred):
            pred = top_num  # Use top_num if prediction is NaN
        # Blend prediction with frequency and historical predictions (e.g., 50% pred, 30% freq, 20% hist)
        hist_weight = 0.2 if hist_freq else 0
        freq_weight = 0.3
        pred_weight = 0.5
        hist_num = hist_sorted[0][0] if hist_sorted else top_num
        blended = int(pred_weight * pred + freq_weight * top_num + hist_weight * hist_num)
        blended = max(1, min(36, blended))  # Clamp to 1-36
        if blended not in used:
            new_seq.append(blended)
            used.add(blended)
            prev = (new_seq + [0] * 9)[:9]  # Update prev with current sequence padded to 9
        else:
            # Try next most frequent not used, prioritizing historical
            candidates = []
            if hist_sorted:
                candidates.extend([num for num, _ in hist_sorted if num not in used])
            freq_sorted = freq.sort_values(ascending=False)
            candidates.extend([num for num in freq_sorted.index if num not in used and num not in candidates])
            if candidates:
                blended = int(candidates[0])
                new_seq.append(blended)
                used.add(blended)
                prev = (new_seq + [0] * 9)[:9]
            else:
                # If all frequent used, pick random available
                available = [n for n in range(1, 37) if n not in used]
                if available:
                    blended = random.choice(available)
                    new_seq.append(blended)
                    used.add(blended)
                    prev = (new_seq + [0] * 9)[:9]
                else:
                    # All numbers used, pick a duplicate (shouldn't happen for 5 out of 36)
                    blended = random.randint(1, 36)
                    new_seq.append(blended)
                    used.add(blended)
                    prev = (new_seq + [0] * 9)[:9]

    # Predict BB and Letter based on the generated sequence
    pred_bb = model_bb.predict(np.array([new_seq]))[0]
    pred_bb = int(max(1, min(36, pred_bb)))  # Clamp BB to reasonable range, assuming 1-36
    pred_letter = model_letter.predict(np.array([new_seq]))[0]

    # Generate SUP6 if model available, ensure no duplicates
    sup6 = None
    if len(X_sup) > 0:
        new_sup_seq = []
        used_sup = set()
        for pos in range(6):
            attempts = 0
            while attempts < 10:  # Limit attempts to prevent infinite loop
                if pos == 0:
                    pred_input = [0] * 6
                else:
                    pred_input = new_sup_seq[:pos] + [0] * (6 - pos)
                pred_sup = float(model_sup.predict(np.array([pred_input]))[0])
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
                    break
                else:
                    # Try next most frequent not used
                    freq_sorted_sup = freq_sup.sort_values(ascending=False)
                    for num in freq_sorted_sup.index:
                        if num not in used_sup:
                            blended_sup = int(num)
                            new_sup_seq.append(blended_sup)
                            used_sup.add(blended_sup)
                            break
                    else:
                        # If all frequent used, pick random available
                        available_sup = [n for n in range(1, 37) if n not in used_sup]
                        if available_sup:
                            blended_sup = random.choice(available_sup)
                            new_sup_seq.append(blended_sup)
                            used_sup.add(blended_sup)
                        break
                attempts += 1
            if attempts >= 10:
                # Fallback: random unique if still not found
                available_sup = [n for n in range(1, 37) if n not in used_sup]
                if available_sup:
                    blended_sup = random.choice(available_sup)
                    new_sup_seq.append(blended_sup)
                    used_sup.add(blended_sup)
        pred_sup_letter = model_sup_letter.predict(np.array([new_seq]))[0]
        sup6 = {'numbers': new_sup_seq, 'letter': pred_sup_letter}

    # Apply Monte Carlo and avoid patterns
    monte_carlo_seq = monte_carlo_simulation(df)
    # Blend with generated sequence
    final_seq = {'numbers': [(a + b) // 2 for a, b in zip(new_seq, monte_carlo_seq)], 'bb': pred_bb, 'letter': pred_letter, 'sup6': sup6}
    final_seq = avoid_popular_patterns(final_seq)

    return final_seq, model, kmeans

@app.route('/add_nla', methods=['POST'])
def add_nla():
    global global_df, global_model, global_kmeans, global_unsaved_changes
    if global_df is None:
        return redirect(url_for('index'))

    draw = request.form.get('draw')
    nla = request.form.get('nla')
    bb = request.form.get('bb')
    letter = request.form.get('letter')

    draw_val = None

    message = None

    # Check for duplicate Draw
    if draw and draw.isdigit():
        draw_val = int(draw)
        if draw_val in global_df['Draw'].dropna().values:
            message = f"Draw {draw_val} is already in the database. Cannot add duplicate draw."
            return render_template('results.html', analysis=analyze_patterns(global_df), sequence={'numbers': [], 'bb': 0, 'letter': 'A', 'sup6': None}, message=message)

    import re
    # Parse NLA, BB, Letter
    if nla and bb and letter:
        numbers = re.findall(r'\d+', nla)
        if len(numbers) == 5:
            nums = [int(n) for n in numbers]
            if all(1 <= n <= 36 for n in nums) and bb.isdigit() and letter.isalpha() and len(letter) == 1:
                bb_val = int(bb)
                letter_val = letter.upper()

                # SUP6 defaults to '00' for missing
                sup_parsed = ['00'] * 6 + ['A']

                new_row = pd.DataFrame([[draw_val, None] + nums + [bb_val, letter_val] + sup_parsed], columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupLetter'])
                global_df = pd.concat([global_df, new_row], ignore_index=True)
                message = "New NLA draw added successfully."
                global_unsaved_changes = True
                save_data_to_csv()

    # Retrain model with updated df
    analysis = analyze_patterns(global_df)
    generated_sequence, model, kmeans = generate_sequence(global_df)
    global_model = model
    global_kmeans = kmeans

    # Calculate last NLA and SUP6, and totals
    last_nla = None
    last_sup6 = None
    total_nla = 0
    total_sup6 = 0

    if not global_df.empty:
        # Last NLA: row with max Draw
        nla_rows = global_df[global_df['Draw'].notna()]
        if not nla_rows.empty:
            last_nla_row = nla_rows.loc[nla_rows['Draw'].idxmax()]
            last_nla = f"Draw {int(last_nla_row['Draw'])}: {int(last_nla_row['Num1'])} {int(last_nla_row['Num2'])} {int(last_nla_row['Num3'])} {int(last_nla_row['Num4'])} {int(last_nla_row['Num5'])} BB: {int(last_nla_row['BB'])} Letter: {last_nla_row['Letter']}"
            total_nla = len(nla_rows)

        # Last SUP6: row with max Draw#
        sup6_rows = global_df[global_df['Draw#'].notna()]
        if not sup6_rows.empty:
            last_sup6_row = sup6_rows.loc[sup6_rows['Draw#'].idxmax()]
            last_sup6 = f"Draw# {int(last_sup6_row['Draw#'])}: {last_sup6_row['Sup1']} {last_sup6_row['Sup2']} {last_sup6_row['Sup3']} {last_sup6_row['Sup4']} {last_sup6_row['Sup5']} {last_sup6_row['Sup6']} Letter: {last_sup6_row['SupLetter']}"
            total_sup6 = len(sup6_rows)

    return render_template('results.html', analysis=analysis, sequence=generated_sequence, message=message, last_nla=last_nla, last_sup6=last_sup6, total_nla=total_nla, total_sup6=total_sup6)

@app.route('/add_sup6', methods=['POST'])
def add_sup6():
    global global_df, global_model, global_kmeans, global_unsaved_changes
    if global_df is None:
        return redirect(url_for('index'))

    draw = request.form.get('draw_num')
    sup6 = request.form.get('sup6')

    message = None

    if draw and draw.isdigit():
        draw_val = int(draw)
        # Check for duplicate Draw#
        if draw_val in global_df['Draw#'].dropna().values:
            message = f"Draw# {draw_val} is already in the database. Cannot add duplicate draw#."
            return render_template('results.html', analysis=analyze_patterns(global_df), sequence={'numbers': [], 'bb': 0, 'letter': 'A', 'sup6': None}, message=message)

        import re
        # Parse SUP6
        if sup6:
            sup6_parts = sup6.split()
            sup6_nums = []
            sup6_letter = 'A'
            for part in sup6_parts:
                if part.isdigit():
                    sup6_nums.append(int(part))
                elif part.isalpha() and len(part) == 1:
                    sup6_letter = part.upper()
            if len(sup6_nums) == 6:
                sup_parsed = sup6_nums + [sup6_letter]

                # Check if draw exists
                if draw_val in global_df['Draw'].values:
                    # Update existing row
                    idx = global_df[global_df['Draw'] == draw_val].index[0]
                    global_df.loc[idx, ['Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupLetter']] = sup_parsed
                    message = f"SUP6 added to existing draw {draw_val}."
                else:
                    # Create new row with NLA as zeros
                    nla_parsed = [0] * 7  # Draw, Draw#, Num1-5, BB, Letter
                    new_row = pd.DataFrame([[None, draw_val] + [0]*5 + [0, 'A'] + sup_parsed], columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupLetter'])
                    global_df = pd.concat([global_df, new_row], ignore_index=True)
                    message = f"New SUP6 draw {draw_val} added successfully."
                global_unsaved_changes = True

                # Retrain model with updated df and regenerate sequence
                analysis = analyze_patterns(global_df)
                generated_sequence, model, kmeans = generate_sequence(global_df)
                global_model = model
                global_kmeans = kmeans
                global_current_sequence = generated_sequence

                # Calculate last NLA and SUP6, and totals
                last_nla = None
                last_sup6 = None
                total_nla = 0
                total_sup6 = 0

                if not global_df.empty:
                    # Last NLA: row with max Draw
                    nla_rows = global_df[global_df['Draw'].notna()]
                    if not nla_rows.empty:
                        last_nla_row = nla_rows.loc[nla_rows['Draw'].idxmax()]
                        last_nla = f"Draw {int(last_nla_row['Draw'])}: {int(last_nla_row['Num1'])} {int(last_nla_row['Num2'])} {int(last_nla_row['Num3'])} {int(last_nla_row['Num4'])} {int(last_nla_row['Num5'])} BB: {int(last_nla_row['BB'])} Letter: {last_nla_row['Letter']}"
                        total_nla = len(nla_rows)

                    # Last SUP6: row with max Draw#
                    sup6_rows = global_df[global_df['Draw#'].notna()]
                    if not sup6_rows.empty:
                        last_sup6_row = sup6_rows.loc[sup6_rows['Draw#'].idxmax()]
                        last_sup6 = f"Draw# {int(last_sup6_row['Draw#'])}: {last_sup6_row['Sup1']} {last_sup6_row['Sup2']} {last_sup6_row['Sup3']} {last_sup6_row['Sup4']} {last_sup6_row['Sup5']} {last_sup6_row['Sup6']} Letter: {last_sup6_row['SupLetter']}"
                        total_sup6 = len(sup6_rows)

                return render_template('results.html', analysis=analysis, sequence=generated_sequence, message=message, last_nla=last_nla, last_sup6=last_sup6, total_nla=total_nla, total_sup6=total_sup6)
            else:
                message = "SUP6 must contain at least 6 numbers."
                return render_template('results.html', analysis=analyze_patterns(global_df), sequence={'numbers': [], 'bb': 0, 'letter': 'A', 'sup6': None}, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)

    return render_template('results.html', analysis=analyze_patterns(global_df), sequence=global_current_sequence, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)

@app.route('/remove_nla', methods=['POST'])
def remove_nla():
    global global_df, global_model, global_kmeans, global_unsaved_changes
    if global_df is None:
        return redirect(url_for('index'))

    draw = request.form.get('draw')
    message = None

    if draw and draw.isdigit():
        draw_val = int(draw)
        if draw_val in global_df['Draw'].dropna().values:
            global_df = global_df[global_df['Draw'] != draw_val]
            message = f"NLA draw {draw_val} removed successfully."
            global_unsaved_changes = True
            save_data_to_csv()

            # Retrain model with updated df
            analysis = analyze_patterns(global_df)
            generated_sequence, model, kmeans = generate_sequence(global_df)
            global_model = model
            global_kmeans = kmeans
            global_current_sequence = generated_sequence

            return render_template('results.html', analysis=analysis, sequence=generated_sequence, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)  # Recalculate in results
        else:
            message = f"NLA draw {draw_val} not found."
    else:
        message = "Please enter a valid draw number."

    return render_template('results.html', analysis=analyze_patterns(global_df), sequence=global_current_sequence, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)

@app.route('/remove_sup6', methods=['POST'])
def remove_sup6():
    global global_df, global_model, global_kmeans, global_unsaved_changes
    if global_df is None:
        return redirect(url_for('index'))

    draw = request.form.get('draw_num')
    message = None

    if draw and draw.isdigit():
        draw_val = int(draw)
        if draw_val in global_df['Draw#'].dropna().values:
            global_df = global_df[global_df['Draw#'] != draw_val]
            message = f"SUP6 draw {draw_val} removed successfully."
            global_unsaved_changes = True
            save_data_to_csv()

            # Retrain model with updated df
            analysis = analyze_patterns(global_df)
            generated_sequence, model, kmeans = generate_sequence(global_df)
            global_model = model
            global_kmeans = kmeans
            global_current_sequence = generated_sequence

            return render_template('results.html', analysis=analysis, sequence=generated_sequence, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)
        else:
            message = f"SUP6 draw {draw_val} not found."
    else:
        message = "Please enter a valid draw number."

    return render_template('results.html', analysis=analyze_patterns(global_df), sequence=global_current_sequence, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)

@app.route('/edit_nla', methods=['POST'])
def edit_nla():
    global global_df, global_model, global_kmeans, global_unsaved_changes
    if global_df is None:
        return redirect(url_for('index'))

    draw = request.form.get('draw')
    nla = request.form.get('nla')
    bb = request.form.get('bb')
    letter = request.form.get('letter')

    message = None

    if draw and draw.isdigit() and nla and bb and letter:
        draw_val = int(draw)
        if draw_val in global_df['Draw'].dropna().values:
            import re
            numbers = re.findall(r'\d+', nla)
            if len(numbers) == 5:
                nums = [int(n) for n in numbers]
                if all(1 <= n <= 36 for n in nums) and bb.isdigit() and letter.isalpha() and len(letter) == 1:
                    bb_val = int(bb)
                    letter_val = letter.upper()

                    idx = global_df[global_df['Draw'] == draw_val].index[0]
                    global_df.loc[idx, ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter']] = nums + [bb_val, letter_val]
                    message = f"NLA draw {draw_val} updated successfully."
                    global_unsaved_changes = True
                    save_data_to_csv()

                    # Retrain model with updated df
                    analysis = analyze_patterns(global_df)
                    generated_sequence, model, kmeans = generate_sequence(global_df)
                    global_model = model
                    global_kmeans = kmeans
                    global_current_sequence = generated_sequence

                    return render_template('results.html', analysis=analysis, sequence=generated_sequence, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)
                else:
                    message = "Invalid NLA numbers, BB, or letter."
            else:
                message = "NLA must contain exactly 5 numbers."
        else:
            message = f"NLA draw {draw_val} not found."
    else:
        message = "Please provide all fields."

    return render_template('results.html', analysis=analyze_patterns(global_df), sequence=global_current_sequence, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)

@app.route('/edit_sup6', methods=['POST'])
def edit_sup6():
    global global_df, global_model, global_kmeans, global_unsaved_changes
    if global_df is None:
        return redirect(url_for('index'))

    draw = request.form.get('draw_num')
    sup6 = request.form.get('sup6')

    message = None

    if draw and draw.isdigit() and sup6:
        draw_val = int(draw)
        if draw_val in global_df['Draw#'].dropna().values:
            import re
            sup6_parts = sup6.split()
            sup6_nums = []
            sup6_letter = 'A'
            for part in sup6_parts:
                if part.isdigit():
                    sup6_nums.append(int(part))
                elif part.isalpha() and len(part) == 1:
                    sup6_letter = part.upper()
            if len(sup6_nums) == 6:
                idx = global_df[global_df['Draw#'] == draw_val].index[0]
                global_df.loc[idx, ['Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupLetter']] = sup6_nums + [sup6_letter]
                message = f"SUP6 draw {draw_val} updated successfully."
                global_unsaved_changes = True
                save_data_to_csv()

                # Retrain model with updated df
                analysis = analyze_patterns(global_df)
                generated_sequence, model, kmeans = generate_sequence(global_df)
                global_model = model
                global_kmeans = kmeans
                global_current_sequence = generated_sequence

                return render_template('results.html', analysis=analysis, sequence=generated_sequence, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)
            else:
                message = "SUP6 must contain exactly 6 numbers."
        else:
            message = f"SUP6 draw {draw_val} not found."
    else:
        message = "Please provide all fields."

    return render_template('results.html', analysis=analyze_patterns(global_df), sequence=global_current_sequence, message=message, last_nla=None, last_sup6=None, total_nla=0, total_sup6=0)

def load_existing_df(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupLetter'])
    df = pd.read_csv(filepath)
    if 'NLA' in df.columns:
        df['NLA'] = df['NLA'].astype(str)
        if 'SUP6' in df.columns:
            df['SUP6'] = df['SUP6'].astype(str)
        # Parse sample format to parsed format
        parsed_data = []
        for _, row in df.iterrows():
            nla = row['NLA']
            sup6 = row.get('SUP6', '')
            draw = row.get('Draw ', None)
            draw_num = row.get('Draw#', None)
            # Parse NLA
            import re
            numbers = re.findall(r'\d+', nla)
            if len(numbers) >= 5:
                nums = [int(n) for n in numbers[:5]]
                bb_match = re.search(r'BB[:\-]\s*(\d+)', nla)
                letter_match = re.search(r'Letter:\s*([A-Z])', nla)
                bb = int(bb_match.group(1)) if bb_match else 1
                letter = letter_match.group(1) if letter_match else 'A'
            else:
                nums = [0]*5
                bb = 1
                letter = 'A'
            # Parse SUP6
            sup_parsed = ['00'] * 6 + ['A']
            if sup6 and isinstance(sup6, str):
                numbers_sup = re.findall(r'\d+', sup6)
                if len(numbers_sup) >= 6:
                    sup_parsed = [n.zfill(2) for n in numbers_sup[:6]]
                    letter_match_sup = re.search(r'Letter:\s*([A-Z])', sup6)
                    if letter_match_sup:
                        sup_parsed.append(letter_match_sup.group(1))
                    else:
                        # Check for letter at end
                        parts = sup6.split()
                        for part in reversed(parts):
                            if part.isalpha() and len(part) == 1:
                                sup_parsed.append(part.upper())
                                break
                        else:
                            sup_parsed.append('A')
                else:
                    sup_parsed = ['00'] * 6 + ['A']
            parsed_data.append([draw, draw_num] + nums + [bb, letter] + sup_parsed)
        return pd.DataFrame(parsed_data, columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupLetter'])
    else:
        # Already in parsed format
        return df

def convert_df_to_sample_format(df):
    new_rows = []
    for _, row in df.iterrows():
        # Construct NLA
        nla = f"{int(row['Num1'])} {int(row['Num2'])} {int(row['Num3'])} {int(row['Num4'])} {int(row['Num5'])} BB: {int(row['BB'])} Letter: {row['Letter']}"
        # Construct SUP6 without BB, format numbers with leading zeros
        sup6 = f"{row['Sup1']} {row['Sup2']} {row['Sup3']} {row['Sup4']} {row['Sup5']} {row['Sup6']} Letter: {row['SupLetter']}"
        # Draw and Draw#
        draw = row['Draw']
        draw_num = row['Draw#']
        new_rows.append({
            'NLA': nla,
            'SUP6': sup6,
            'Draw ': draw,
            'Draw#': draw_num
        })
    return pd.DataFrame(new_rows)

def save_data_to_csv():
    global global_df, global_unsaved_changes
    print(f"save_data_to_csv called: global_df is None: {global_df is None}")
    if global_df is not None:
        try:
            df_to_save = convert_df_to_sample_format(global_df)
            print(f"Saving {len(df_to_save)} rows to {SAVE_PATH}")
            df_to_save.to_csv(SAVE_PATH, index=False)
            global_unsaved_changes = False
            print("Data saved successfully.")
            return True, "Data saved successfully."
        except Exception as e:
            print(f"Error saving to {SAVE_PATH}: {e}")
            return False, f"Error saving data: {str(e)}"
    print("No data to save.")
    return False, "No data to save."

def save_prediction_to_history(sequence):
    import datetime
    history_file = 'prediction_history.csv'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    nla = '-'.join(map(str, sequence['numbers'])) + f' BB: {sequence["bb"]} Letter: {sequence["letter"]}'
    sup6_str = ''
    if sequence.get('sup6'):
        sup6 = sequence['sup6']
        sup6_str = '-'.join(map(str, sup6['numbers'])) + f' Letter: {sup6["letter"]}'
    
    new_row = pd.DataFrame({
        'timestamp': [timestamp],
        'nla_prediction': [nla],
        'sup6_prediction': [sup6_str]
    })
    
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        df_hist = pd.concat([new_row, df_hist], ignore_index=True)
    else:
        df_hist = new_row
    
    df_hist.to_csv(history_file, index=False)

@app.route('/results')
def results():
    global global_df, global_model, global_kmeans, global_current_sequence, global_csv_path
    # Load saved data if not loaded
    if global_df is None:
        global_df = load_saved_data()
        if global_df is not None:
            # Set global_csv_path to SAVE_PATH
            global_csv_path = SAVE_PATH
            # Generate sequence if loaded
            analysis = analyze_patterns(global_df)
            generated_sequence, model, kmeans = generate_sequence(global_df)
            global_model = model
            global_kmeans = kmeans
            global_current_sequence = generated_sequence
    if global_df is None or global_df.empty:
        return redirect(url_for('index'))
    analysis = analyze_patterns(global_df)

    # Calculate last NLA and SUP6, and totals
    last_nla = None
    last_sup6 = None
    total_nla = 0
    total_sup6 = 0

    if not global_df.empty:
        # Last NLA: row with max Draw
        nla_rows = global_df[global_df['Draw'].notna()]
        if not nla_rows.empty:
            last_nla_row = nla_rows.loc[nla_rows['Draw'].idxmax()]
            last_nla = f"Draw {int(last_nla_row['Draw'])}: {int(last_nla_row['Num1'])} {int(last_nla_row['Num2'])} {int(last_nla_row['Num3'])} {int(last_nla_row['Num4'])} {int(last_nla_row['Num5'])} BB: {int(last_nla_row['BB'])} Letter: {last_nla_row['Letter']}"
            total_nla = len(nla_rows)

        # Last SUP6: row with max Draw#
        sup6_rows = global_df[global_df['Draw#'].notna()]
        if not sup6_rows.empty:
            last_sup6_row = sup6_rows.loc[sup6_rows['Draw#'].idxmax()]
            last_sup6 = f"Draw# {int(last_sup6_row['Draw#'])}: {last_sup6_row['Sup1']} {last_sup6_row['Sup2']} {last_sup6_row['Sup3']} {last_sup6_row['Sup4']} {last_sup6_row['Sup5']} {last_sup6_row['Sup6']} Letter: {last_sup6_row['SupLetter']}"
            total_sup6 = len(sup6_rows)

    return render_template('results.html', analysis=analysis, sequence=global_current_sequence, last_nla=last_nla, last_sup6=last_sup6, total_nla=total_nla, total_sup6=total_sup6)

@app.route('/search', methods=['GET', 'POST'])
def search():
    global global_df, global_model, global_kmeans, global_current_sequence
    results = None
    message = None
    draw = None
    if request.method == 'POST':
        draw = request.form.get('draw')
        if draw and draw.isdigit():
            draw_val = int(draw)
            # Load saved data if not loaded
            if global_df is None:
                global_df = load_saved_data()
                if global_df is not None:
                    # Generate sequence if loaded
                    analysis = analyze_patterns(global_df)
                    generated_sequence, model, kmeans = generate_sequence(global_df)
                    global_model = model
                    global_kmeans = kmeans
                    global_current_sequence = generated_sequence
            if global_df is not None and not global_df.empty:
                # Search for NLA by Draw
                nla_row = global_df[global_df['Draw'] == draw_val]
                if not nla_row.empty:
                    row = nla_row.iloc[0]
                    nla = f"{int(row['Num1'])} {int(row['Num2'])} {int(row['Num3'])} {int(row['Num4'])} {int(row['Num5'])} BB: {int(row['BB'])} Letter: {row['Letter']}"
                else:
                    nla = None

                # Search for SUP6 by Draw#
                sup6_row = global_df[global_df['Draw#'] == draw_val]
                if not sup6_row.empty:
                    row = sup6_row.iloc[0]
                    sup6 = f"{row['Sup1']} {row['Sup2']} {row['Sup3']} {row['Sup4']} {row['Sup5']} {row['Sup6']} Letter: {row['SupLetter']}"
                else:
                    sup6 = None

                if nla or sup6:
                    results = {'nla': nla, 'sup6': sup6}
                else:
                    message = f"No data found for Draw {draw_val}."
            else:
                message = "No data uploaded yet. Please upload a CSV first."
        else:
            message = "Please enter a valid draw number."
    global_df_exists = global_df is not None and not global_df.empty
    return render_template('search.html', results=results, message=message, draw=draw, global_df_exists=global_df_exists)

@app.route('/history')
def history():
    history_file = 'prediction_history.csv'
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        # Remove duplicates based on nla_prediction to improve user viewing
        df_hist = df_hist.drop_duplicates(subset=['nla_prediction'], keep='first')
        history_list = df_hist.to_dict('records')
    else:
        history_list = []
    return render_template('history.html', history=history_list)

@app.route('/save', methods=['POST'])
def save():
    global global_df, global_unsaved_changes, global_csv_path
    if global_df is not None:
        try:
            save_path = global_csv_path if global_csv_path else r'C:\Users\jomal\OneDrive\Documents\reconstructed_lotto.csv'
            convert_df_to_sample_format(global_df).to_csv(save_path, index=False)
            global_unsaved_changes = False
            return {'status': 'success', 'message': 'Data saved successfully.'}
        except Exception as e:
            return {'status': 'error', 'message': f'Error saving data: {str(e)}'}
    return {'status': 'error', 'message': 'No data to save.'}

@app.route('/download')
def download():
    global global_csv_path
    if global_csv_path and os.path.exists(global_csv_path):
        return send_file(global_csv_path, as_attachment=True, download_name='updated_lotto_data.csv')
    return "No file available for download."

@app.route('/exit')
def exit_app():
    return render_template('exit.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)

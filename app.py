
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Global variables to store data and model
global_df = None
global_model = None
global_kmeans = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Load and validate data
            df = pd.read_csv(filepath)
            if 'NLA' not in df.columns:
                return "Error: CSV must have a column named 'NLA'."
            if len(df) < 10:
                return "Error: CSV must have at least 10 rows."

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
                if not all(1 <= n <= 36 for n in nums):
                    return f"Error: Numbers in '{nla}' must be between 1 and 36."
                # Extract BB and Letter, add defaults if missing
                bb_match = re.search(r'BB[:\-]\s*(\d+)', nla)
                letter_match = re.search(r'Letter:\s*([A-Z])', nla)
                bb = int(bb_match.group(1)) if bb_match else 1
                letter = letter_match.group(1) if letter_match else 'A'

                # Parse SUP6 if present
                sup_parsed = [0] * 8  # 6 nums + bb + letter
                if 'SUP6' in df.columns and not pd.isna(row['SUP6']) and isinstance(row['SUP6'], str):
                    sup6 = row['SUP6']
                    numbers_sup = re.findall(r'\d+', sup6)
                    if len(numbers_sup) >= 6:
                        nums_sup = [int(n) for n in numbers_sup[:6]]
                        if all(1 <= n <= 36 for n in nums_sup):
                            bb_match_sup = re.search(r'BB[:\-]\s*(\d+)', sup6)
                            letter_match_sup = re.search(r'Letter:\s*([A-Z])', sup6)
                            bb_sup = int(bb_match_sup.group(1)) if bb_match_sup else 1
                            letter_sup = letter_match_sup.group(1) if letter_match_sup else 'A'
                            sup_parsed = nums_sup + [bb_sup, letter_sup]

                # Include Draw and Draw# if available
                draw_val = row.get('Draw ', None)
                draw_num_val = row.get('Draw#', None)
                parsed_data.append([draw_val, draw_num_val] + nums + [bb, letter] + sup_parsed)
            df = pd.DataFrame(parsed_data, columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupBB', 'SupLetter'])

            # Store globally for recalibration
            global global_df, global_model, global_kmeans
            global_df = df.copy()

            # Analysis
            analysis = analyze_patterns(df)

            # ML and Generation
            generated_sequence, model, kmeans = generate_sequence(df)
            global_model = model
            global_kmeans = kmeans

            # Save prediction to history
            save_prediction_to_history(generated_sequence)

            return render_template('results.html', analysis=analysis, sequence=generated_sequence)
    return render_template('index.html')

def analyze_patterns(df):
    results = {}

    # Positional frequencies
    for pos in range(5):
        results[f'Position {pos+1} Frequencies'] = df.iloc[:, pos].value_counts().to_dict()

    # Numbers never called
    all_numbers = set(range(1, 37))
    used_numbers = set(df.values.flatten())
    results['Never Called'] = list(all_numbers - used_numbers)

    # Numbers always called (in all 114 sequences)
    always_called = set(range(1, 37))
    for _, row in df.iterrows():
        always_called &= set(row)
    results['Always Called'] = list(always_called) if always_called else "None"

    # Positional repeats (how often a number repeats in the same position across sequences)
    repeats = {}
    for pos in range(5):
        pos_data = df.iloc[:, pos]
        repeats[f'Position {pos+1}'] = pos_data.value_counts().max()  # Max count for any number
    results['Max Repeats per Position'] = repeats

    return results

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
            y.append(next_seq[pos+2])  # Target: next position's number (Num1 is column 2)
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
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # Train model for BB
    model_bb = RandomForestRegressor(n_estimators=100)
    model_bb.fit(X_bb_letter, y_bb)

    # Train model for Letter
    model_letter = RandomForestClassifier(n_estimators=100)
    model_letter.fit(X_bb_letter, y_letter)

    # Train SUP6 models if data available
    if len(X_sup) > 0:
        model_sup = RandomForestRegressor(n_estimators=100)
        model_sup.fit(X_sup, y_sup)
        model_sup_bb = RandomForestRegressor(n_estimators=100)
        model_sup_bb.fit(X_sup_bb_letter, y_sup_bb)
        model_sup_letter = RandomForestClassifier(n_estimators=100)
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

@app.route('/add_nla', methods=['POST'])
def add_nla():
    global global_df, global_model, global_kmeans
    if global_df is None:
        return redirect(url_for('index'))

    draw = request.form.get('draw')
    nla = request.form.get('nla')
    bb = request.form.get('bb')
    letter = request.form.get('letter')
    draw_num = request.form.get('draw_num')

    message = None

    # Check for duplicate Draw or Draw#
    if draw and draw.isdigit():
        draw_val = int(draw)
        if draw_val in global_df['Draw'].values:
            message = f"Draw {draw_val} is already in the database. Cannot add duplicate draw."
            return render_template('results.html', analysis=analyze_patterns(global_df), sequence={'numbers': [], 'bb': 0, 'letter': 'A', 'sup6': {}}, message=message)

    if draw_num and draw_num.isdigit():
        draw_num_val = int(draw_num)
        if draw_num_val in global_df['Draw#'].values:
            message = f"Draw# {draw_num_val} is already in the database. Cannot add duplicate draw#."
            return render_template('results.html', analysis=analyze_patterns(global_df), sequence={'numbers': [], 'bb': 0, 'letter': 'A', 'sup6': {}}, message=message)

    import re
    # Parse NLA, BB, Letter
    if nla and bb and letter:
        numbers = re.findall(r'\d+', nla)
        if len(numbers) == 5:
            nums = [int(n) for n in numbers]
            if all(1 <= n <= 36 for n in nums) and bb.isdigit() and letter.isalpha() and len(letter) == 1:
                bb_val = int(bb)
                letter_val = letter.upper()

                # SUP6 defaults to zeros
                sup_parsed = [0] * 8

                new_row = pd.DataFrame([[draw_val, draw_num_val] + nums + [bb_val, letter_val] + sup_parsed], columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupBB', 'SupLetter'])
                global_df = pd.concat([global_df, new_row], ignore_index=True)
                message = "New NLA draw added successfully."

    # Retrain model with updated df
    analysis = analyze_patterns(global_df)
    generated_sequence, model, kmeans = generate_sequence(global_df)
    global_model = model
    global_kmeans = kmeans

    return render_template('results.html', analysis=analysis, sequence=generated_sequence, message=message)

@app.route('/add_sup6', methods=['POST'])
def add_sup6():
    global global_df, global_model, global_kmeans
    if global_df is None:
        return redirect(url_for('index'))

    draw = request.form.get('draw_num')
    sup6 = request.form.get('sup6')

    message = None

    if draw and draw.isdigit():
        draw_val = int(draw)
        import re
        # Parse SUP6
        if sup6:
            sup6_parts = sup6.split()
            sup6_nums = []
            sup6_bb = 1
            sup6_letter = 'A'
            for part in sup6_parts:
                if part.isdigit():
                    sup6_nums.append(int(part))
                elif part.isalpha() and len(part) == 1:
                    sup6_letter = part.upper()
            if len(sup6_nums) >= 6:
                sup6_nums = sup6_nums[:6]
                # Extract BB if present in sup6 string
                bb_match_sup = re.search(r'BB[:\-]\s*(\d+)', sup6)
                if bb_match_sup:
                    sup6_bb = int(bb_match_sup.group(1))
                sup_parsed = sup6_nums + [sup6_bb, sup6_letter]

                # Check if draw exists
                if draw_val in global_df['Draw'].values:
                    # Update existing row
                    idx = global_df[global_df['Draw'] == draw_val].index[0]
                    global_df.loc[idx, ['Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupBB', 'SupLetter']] = sup_parsed
                    message = f"SUP6 added to existing draw {draw_val}."
                else:
                    # Create new row with NLA as zeros
                    nla_parsed = [0] * 7  # Draw, Draw#, Num1-5, BB, Letter
                    new_row = pd.DataFrame([[draw_val, None] + [0]*5 + [0, 'A'] + sup_parsed], columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupBB', 'SupLetter'])
                    global_df = pd.concat([global_df, new_row], ignore_index=True)
                    message = f"New SUP6 draw {draw_val} added successfully."
            else:
                message = "SUP6 must contain at least 6 numbers."
                return render_template('results.html', analysis=analyze_patterns(global_df), sequence={'numbers': [], 'bb': 0, 'letter': 'A', 'sup6': {}}, message=message)

    # Retrain model with updated df
    analysis = analyze_patterns(global_df)
    generated_sequence, model, kmeans = generate_sequence(global_df)
    global_model = model
    global_kmeans = kmeans

    return render_template('results.html', analysis=analysis, sequence=generated_sequence, message=message)

def save_prediction_to_history(sequence):
    import datetime
    history_file = 'prediction_history.csv'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    nla = '-'.join(map(str, sequence['numbers'])) + f' BB: {sequence["bb"]} Letter: {sequence["letter"]}'
    sup6_str = ''
    if sequence.get('sup6'):
        sup6 = sequence['sup6']
        sup6_str = '-'.join(map(str, sup6['numbers'])) + f' BB: {sup6["bb"]} Letter: {sup6["letter"]}'
    
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

@app.route('/history')
def history():
    history_file = 'prediction_history.csv'
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        history_list = df_hist.to_dict('records')
    else:
        history_list = []
    return render_template('history.html', history=history_list)

if __name__ == '__main__':
    app.run(debug=True, port=3000)

from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

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

                parsed_data.append(nums + [bb, letter] + sup_parsed)
            df = pd.DataFrame(parsed_data, columns=['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupBB', 'SupLetter'])

            # Store globally for recalibration
            global global_df, global_model, global_kmeans
            global_df = df.copy()

            # Analysis
            analysis = analyze_patterns(df)

            # ML and Generation
            generated_sequence, model, kmeans = generate_sequence(df)
            global_model = model
            global_kmeans = kmeans

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
            y.append(next_seq[pos])  # Target: next position's number
        # For BB and Letter, use the full previous sequence as features
        X_bb_letter.append(seq)
        y_bb.append(next_seq['BB'])
        y_letter.append(next_seq['Letter'])

        # For SUP6, if available
        if next_seq['Sup1'] != 0:  # Assuming 0 means not parsed
            for pos in range(6):
                X_sup.append((seq + [0] * (6 - len(seq)))[:6])  # Pad to 6 for SUP6 features
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

    # Generate new sequence: Start with a random seed, predict step-by-step, bias toward high-freq numbers
    new_seq = []
    prev = [0] * 5  # Start with padded features
    for pos in range(5):
        pred = model.predict([prev])[0]
        # Bias: Adjust toward most frequent in that position
        freq = df.iloc[:, pos].value_counts()
        top_num = freq.idxmax()
        # Blend prediction with frequency (e.g., 70% pred, 30% top freq)
        blended = int(0.7 * pred + 0.3 * top_num)
        blended = max(1, min(36, blended))  # Clamp to 1-36
        new_seq.append(blended)
        prev = new_seq + [0] * (5 - len(new_seq))  # Update prev with current sequence padded

    # Predict BB and Letter based on the generated sequence
    pred_bb = model_bb.predict([new_seq])[0]
    pred_bb = int(max(1, min(36, pred_bb)))  # Clamp BB to reasonable range, assuming 1-36
    pred_letter = model_letter.predict([new_seq])[0]

    # Generate SUP6 if model available
    sup6 = {}
    if len(X_sup) > 0:
        new_sup_seq = []
        prev_sup = [0] * 6
        for pos in range(6):
            pred_sup = model_sup.predict([prev_sup])[0]
            # Bias toward freq
            freq_sup = df.iloc[:, 7 + pos].value_counts()  # Sup1 to Sup6 are columns 7 to 12
            top_num_sup = freq_sup.idxmax()
            blended_sup = int(0.7 * pred_sup + 0.3 * top_num_sup)
            blended_sup = max(1, min(36, blended_sup))
            new_sup_seq.append(blended_sup)
            prev_sup = new_sup_seq + [0] * (6 - len(new_sup_seq))
        pred_sup_bb = model_sup_bb.predict([new_seq])[0]
        pred_sup_bb = int(max(1, min(36, pred_sup_bb)))
        pred_sup_letter = model_sup_letter.predict([new_seq])[0]
        sup6 = {'numbers': new_sup_seq, 'bb': pred_sup_bb, 'letter': pred_sup_letter}

    return {'numbers': new_seq, 'bb': pred_bb, 'letter': pred_letter, 'sup6': sup6}, model, kmeans

@app.route('/add_numbers', methods=['POST'])
def add_numbers():
    global global_df, global_model, global_kmeans
    if global_df is None:
        return redirect(url_for('index'))

    draw = request.form.get('draw')
    nla = request.form.get('nla')
    bb = request.form.get('bb')
    letter = request.form.get('letter')
    draw_num = request.form.get('draw_num')
    sup6 = request.form.get('sup6')

    import re
    # Parse NLA, BB, Letter
    if nla and bb and letter:
        numbers = re.findall(r'\d+', nla)
        if len(numbers) == 5:
            nums = [int(n) for n in numbers]
            if all(1 <= n <= 36 for n in nums) and bb.isdigit() and letter.isalpha() and len(letter) == 1:
                bb_val = int(bb)
                letter_val = letter.upper()
                new_row = pd.DataFrame([nums + [bb_val, letter_val]], columns=['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter'])
                global_df = pd.concat([global_df, new_row], ignore_index=True)

    # Parse SUP6 (optional, but if provided, could be used for additional analysis if needed)
    if sup6:
        sup6_parts = sup6.split()
        sup6_nums = []
        sup6_letter = None
        for part in sup6_parts:
            if part.isdigit():
                sup6_nums.append(int(part))
            elif part.isalpha() and len(part) == 1:
                sup6_letter = part.upper()
        if len(sup6_nums) == 6 and all(1 <= n <= 36 for n in sup6_nums):
            # For now, just acknowledge it, but since the model is based on NLA, we don't append SUP6 to df
            pass

    # Retrain model with updated df
    analysis = analyze_patterns(global_df)
    generated_sequence, model, kmeans = generate_sequence(global_df)
    global_model = model
    global_kmeans = kmeans

    return render_template('results.html', analysis=analysis, sequence=generated_sequence)

if __name__ == '__main__':
    app.run(debug=True, port=3000)

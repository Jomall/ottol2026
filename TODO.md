# TODO: Implement Exit Feature with Save Prompt

## Step 1: Add Global Variable for Unsaved Changes
- [x] Add `global_unsaved_changes = False` to app.py globals.

## Step 2: Modify Add/Remove/Edit Routes to Track Changes
- [x] In `/add_nla`, `/add_sup6`, `/remove_nla`, `/remove_sup6`, `/edit_nla`, `/edit_sup6` routes:
  - [x] Set `global_unsaved_changes = True`.
  - [x] Remove the immediate `convert_df_to_sample_format(global_df).to_csv(global_csv_path, index=False)` calls.

## Step 3: Add Save Route
- [x] Add `/save` route in app.py:
  - [x] Save data to CSV using `convert_df_to_sample_format(global_df).to_csv(save_path, index=False)` where save_path is `C:\Users\jomal\OneDrive\Documents\reconstructed_lotto.csv`.
  - [x] Set `global_unsaved_changes = False`.
  - [x] Return success message or redirect.

## Step 4: Add Exit Route
- [x] Add `/exit` route in app.py:
  - [x] Redirect to a goodbye page.

## Step 5: Create Exit Template
- [x] Create `templates/exit.html` with a goodbye message.

## Step 6: Update Results Template
- [x] Add "Exit" button to `results.html`.
- [x] Add JavaScript for modal to prompt save or exit without save.
- [x] On save, call `/save` via fetch, then redirect to `/exit`.
- [x] On exit without save, redirect to `/exit`.

## Step 7: Update Other Templates
- [x] Add "Exit" button to `search.html` with modal and JavaScript.

## Step 8: Test the Feature
- [x] App runs successfully on http://127.0.0.1:3000.
- [x] Fixed ML error in generate_sequence function for SUP6 prediction.
- [x] Save route returns appropriate messages when no data is loaded.
- [x] Data will be saved to C:\Users\jomal\OneDrive\Documents\reconstructed_lotto.csv when save is called.
- [x] Exit feature implemented with prompt to save new uploaded data before closing.

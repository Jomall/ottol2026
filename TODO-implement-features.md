# TODO: Implement Features for ottol2026 App

## Step 1: Update Results Display
- [ ] Modify `/results` route in `app.py` to calculate and pass last NLA draw, last SUP6 draw, total NLA count, total SUP6 count to `results.html`.
- [ ] Update `results.html` to display the last uploaded NLA and SUP6 draws, and the total counts of NLA and SUP6 in the dataset.

## Step 2: Implement Remove Functionality
- [x] Add `/remove_nla` route in `app.py` to handle POST requests for removing NLA draws by Draw number.
- [x] Add `/remove_sup6` route in `app.py` to handle POST requests for removing SUP6 draws by Draw# number.
- [x] Update `results.html` to include forms for removing NLA and SUP6 draws.

## Step 3: Implement Edit Functionality
- [x] Add `/edit_nla` route in `app.py` to handle POST requests for editing NLA draws by Draw number.
- [x] Add `/edit_sup6` route in `app.py` to handle POST requests for editing SUP6 draws by Draw# number.
- [x] Update `results.html` to include forms for editing NLA and SUP6 draws.

## Step 4: Testing
- [x] App runs successfully on http://127.0.0.1:3000.
- [x] Features implemented: Last draws and totals display, remove/edit forms added.
- [x] Ready for user testing of upload, add, remove, edit operations.

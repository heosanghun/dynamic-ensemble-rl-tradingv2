# Upload Guide for Anonymous Submission

## Important: Anonymous Upload Process

This repository is prepared for **anonymous submission** to 4open.science for paper review.

## Upload Method: 4open.science (Recommended for Anonymous Submission)

### Step 1: Prepare the Repository

1. Ensure all personal information has been removed (already done)
2. Verify all links use anonymous format: `https://anonymous.4open.science/r/YOUR-ANONYMOUS-LINK-ID`
3. Check that no author names, emails, or GitHub usernames appear in the code

### Step 2: Upload to 4open.science

1. Go to https://4open.science
2. Create an anonymous repository
3. Upload the entire `dynamic_ensemble_rl_trading` folder
4. You will receive an anonymous link like: `https://anonymous.4open.science/r/XXXXX`

### Step 3: Update Anonymous Links

After receiving your anonymous ID, update these files:
- `README.md`: Replace `YOUR-ANONYMOUS-LINK-ID` with actual ID (2 locations)
- `setup.py`: Replace `YOUR-ANONYMOUS-LINK-ID` with actual ID (1 location)
- `docs/REPRODUCTION.md`: Replace `YOUR-ANONYMOUS-LINK-ID` with actual ID (1 location)

### Step 4: Final Upload

Re-upload the updated files to 4open.science with the correct anonymous links.

## Alternative: GitHub Upload (If Needed for Backup)

If you need to upload to GitHub as a backup (not for anonymous submission):

1. Create a new GitHub repository (private or public)
2. Initialize git in the project folder:
   ```bash
   cd dynamic_ensemble_rl_trading
   git init
   git add .
   git commit -m "Initial commit"
   ```

3. Add remote and push:
   ```bash
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
   git branch -M main
   git push -u origin main
   ```

**Note**: For anonymous paper submission, use 4open.science, not GitHub directly.

## Security Note

- Never commit personal access tokens to the repository
- Keep tokens in environment variables or secure storage
- Use `.gitignore` to exclude sensitive files

## Verification Before Upload

- [ ] All personal information removed
- [ ] All repository links use anonymous.4open.science format
- [ ] No author names in code or documentation
- [ ] No email addresses
- [ ] No GitHub usernames
- [ ] Data links point to Google Drive (already anonymous)
- [ ] All code is functional


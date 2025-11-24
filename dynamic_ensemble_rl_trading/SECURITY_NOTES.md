# Security Notes

## Important: Token Security

**NEVER commit personal access tokens or API keys to the repository.**

### Token Storage

If you need to use the GitHub token for backup or other purposes:

1. Store tokens in environment variables:
   ```bash
   # Windows PowerShell
   $env:GITHUB_TOKEN="your-token-here"
   
   # Linux/Mac
   export GITHUB_TOKEN="your-token-here"
   ```

2. Or use a local config file (not tracked by git):
   - Create `local_config.json` (already in .gitignore)
   - Never commit this file

### For Anonymous Upload

**Primary Method: Use 4open.science**

For anonymous paper submission, upload directly to 4open.science:
- Go to https://4open.science
- Upload the `dynamic_ensemble_rl_trading` folder
- Receive anonymous link
- Update links in README.md, setup.py, and docs/REPRODUCTION.md

**Do NOT use GitHub for anonymous submission.**

### If Using GitHub (Backup Only)

If you need GitHub as a backup (not for anonymous submission):

1. Create a private repository
2. Use token via environment variable or Git credential helper
3. Never hardcode tokens in code or config files

### Files Protected by .gitignore

The following are automatically excluded:
- `.env` files
- `*.token` files
- `*_token.txt` files
- `secrets.json`
- `config.json`
- `.git/config` (local git config)

## Verification

Before any upload, verify:
- [ ] No tokens in any committed files
- [ ] No API keys in code
- [ ] .gitignore includes all sensitive file patterns
- [ ] Environment variables used for secrets


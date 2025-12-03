1. do not use .get if we know the key should be there. let it fail if a key/value combo is incorrectly not there
- example of bad code:
```py
CONFIG.get('supplier_normalization', {}).get('replacements', {})
```
- reasoning: hard to read, causes issues when we accidentally are missing an important value and are now getting incorrect default values

2. do not create fake data under any circumstances. if for some reason yyou think it would be beneficial, you need to ask me before coding it.
- this is dangerous
- example: you can't connect to an API so you hard-code fake data, or posts/tweets, etc.

3. do not use logic like this for a python script to add the path to src. 
- example:
```py
sys.path.append(str(Path(__file__).parent.parent / 'src'))
```
- better: using os, finding the root of the dir (use config if you can, otherwise find a .gitignore file), 
- reasoning: if you are to drag/drop into a notebook, this causes an error

4. do no check for existence of an item that should be there
- better: check it exists, if not, make code fail

```py
# exmaple 1
if df and df['important_column_name'] and df in globals():
    ...

# example 2
if 'kelly_info' in reasoning and reasoning['kelly_info']:
    ...
```
- reasoning:
    - super hard to read, tons of indenting that is not necessary
    - it is okay to write code that breaks, doing checks such as if a necessary column exists would just be prolonging the failure of the script later on 

5. do NOT search for columns that might work, read the file in, print column names and values, and figure it out for yourself (or ask me!)

```py
# bad
possible_names = ["store_number", "store_nbr", "StoreNumber", "storeNum", "StoNum"]
store_col = next((g for g in possible_names if g in reader.fieldnames), None)

if not store_col:
    print("Couldn't find a store column — maybe it's named something else?")
else:
    print(f"Found store column: {store_col}")
    for row in rows[:5]:
        print(row[store_col])

```

6. when creating Python files from my direction, put the context into the docstring so it is easy for both me AND you to remember what i've asked for
- bad: python file with no docstring + separate .md file
- good: all in 1 python file

7. if you are going to use emojis, have an emoji map in the config or something similar
- good:

```py

# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    # Status
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'refresh': '🔄',
    'save': '💾',
    
    # Analysis
    'chart': '📊',
    'target': '🎯',
    'calendar': '📅',
    'star': '⭐',
    'money': '💰',
    'up': '📈',
    'down': '📉',
    
    # Luck categories
    'lucky': '🍀',
    'unlucky': '💔',
    'neutral': '😐',
    
    # Sports
    'nba': '🏀',
    'nfl': '🏈',
    
    # Test/Debug
    'test': '🧪',
}

```

- bad: hard-coding

8. do not do a git commit / push until you know we have fixed the issue / iterated on the task successfully

9. if you are going to do git commits, ask me for permission first, and use the methodology from this link: 
- Jiawei's article on submitting Git Patches / commit messages
- link: https://git.kernel.org/pub/scm/git/git.git/tree/Documentation/SubmittingPatches

advice on how to make the message:
- Keep it ~50 chars, no period at the end.
- Start with a scope (cli, docs, parser, build, a file/dir).
- Use imperative mood: “add”, “fix”, “refactor”, “revert”, “docs”.
- Say what changed and hint at why/impact (crash avoided, speed up, UX).
- Avoid fluff: no “This commit…”, no emojis, no trailing punctuation.
- Prefer specific nouns over vague ones (“nil ptr on empty input” > “issue”).
- If reverting, start with revert: …. If following up, consider follow-up:.
- Reference an issue only if your team’s convention requires it (short form).
- If it’s a sweeping change, split commits instead; one logical change per commit.

Examples you can copy and tweak:
- git commit --message "parser: fix nil ptr on empty input"
- git commit --message "cli: add --dry-run flag for safety"
- git commit --message "docs: clarify DCO vs PGP signing"
- git commit --message "build: cache deps to cut CI time"

10. 


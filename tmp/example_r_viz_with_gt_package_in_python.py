"""
Quick test to verify R gt package works with fake data.
"""

import pandas as pd
from pathlib import Path

# Create fake data
df = pd.DataFrame({
    'RANK': [1, 2, 3],
    'PLAYER': ['Austin Reaves', 'Joel Embiid', 'Stephen Curry'],
    'GAMES': [3, 5, 7],
    'PPG OVER EXP': ['+12.8', '+6.2', '+4.9'],
    'COVER RATE': ['100.0%', '66.7%', '60.0%'],
    'ppg_over_exp_value': [12.8, 6.2, 4.9]
})

print("Test data:")
print(df)
print()

try:
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    
    print("✅ rpy2 imported successfully")
    
    # Convert to R
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(df)
    
    robjects.globalenv['test_data'] = r_df
    print("✅ Data converted to R successfully")
    
    # Try to create gt table
    user_lib_path = str(Path.home() / 'R' / 'library')
    print(f"📁 R library path: {user_lib_path}")
    
    r_code = f"""
    # Set library path
    .libPaths(c("{user_lib_path}", .libPaths()))
    
    print("R library paths:")
    print(.libPaths())
    
    # Load gt
    library(gt)
    library(gtExtras)
    library(webshot2)
    
    print("✅ Packages loaded successfully")
    
    # Create simple table
    tbl <- test_data %>%
      gt() %>%
      tab_header(
        title = md("**Test Table**"),
        subtitle = "Verify R gt package works"
      ) %>%
      cols_hide(columns = ppg_over_exp_value) %>%
      gt_theme_538()
    
    # Save
    gtsave(tbl, "test_gt_output.png", vwidth = 800, vheight = 400)
    
    print("✅ Table saved successfully")
    """
    
    print("\n🎨 Executing R code...")
    robjects.r(r_code)
    
    print("\n✅ SUCCESS! Check test_gt_output.png")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

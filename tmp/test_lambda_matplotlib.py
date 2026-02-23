"""
Test Lambda function to check if matplotlib is available.

Usage:
    1. Copy this to AWS Lambda console (create new function "test-matplotlib")
    2. Add your pandas-matplotlib-numpy-combined layer
    3. Test with empty event {}
    4. Check CloudWatch logs for results
"""

import json
import logging
import sys

def lambda_handler(event, context):
    """Test if matplotlib and pandas are available."""
    
    results = {
        'python_version': sys.version,
        'sys_path': sys.path,
        'tests': {}
    }
    
    # Test 1: Import pandas
    try:
        import pandas as pd
        results['tests']['pandas'] = {
            'available': True,
            'version': pd.__version__,
            'location': pd.__file__
        }
    except ImportError as e:
        results['tests']['pandas'] = {
            'available': False,
            'error': str(e)
        }
    
    # Test 2: Import numpy
    try:
        import numpy as np
        results['tests']['numpy'] = {
            'available': True,
            'version': np.__version__,
            'location': np.__file__
        }
    except ImportError as e:
        results['tests']['numpy'] = {
            'available': False,
            'error': str(e)
        }
    
    # Test 3: Import matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        results['tests']['matplotlib'] = {
            'available': True,
            'version': matplotlib.__version__,
            'location': matplotlib.__file__
        }
    except ImportError as e:
        results['tests']['matplotlib'] = {
            'available': False,
            'error': str(e)
        }
    
    # Summary
    all_available = all(t.get('available', False) for t in results['tests'].values())
    results['summary'] = {
        'all_packages_available': all_available,
        'message': '✅ All packages available!' if all_available else '❌ Some packages missing'
    }
    
    return {
        'statusCode': 200 if all_available else 500,
        'body': json.dumps(results, indent=2)
    }

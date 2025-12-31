#!/usr/bin/env python3
import re
import sys

def extract_function_body(filepath):
    """Extract function body, skipping #include lines but keeping everything else."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    result = []
    for line in lines:
        # Skip #include lines
        if re.match(r'^\s*#include\s', line):
            continue
        # Keep everything else
        result.append(line.rstrip('\n'))
    
    # Remove leading/trailing empty lines
    while result and not result[0].strip():
        result.pop(0)
    while result and not result[-1].strip():
        result.pop()
    
    return '\n'.join(result)

# Append x_solve, y_solve, z_solve to bt.c
bt_file = 'bt-serial/bt.c'
files_to_append = ['bt-serial/x_solve.c', 'bt-serial/y_solve.c', 'bt-serial/z_solve.c']

# Read current bt.c
with open(bt_file, 'r') as f:
    bt_content = f.read()

# Append each file
for filepath in files_to_append:
    func_body = extract_function_body(filepath)
    bt_content += '\n\n' + func_body

# Write back
with open(bt_file, 'w') as f:
    f.write(bt_content)

print("Successfully appended x_solve, y_solve, z_solve to bt.c")

#!/usr/bin/env python3
import re
import os
import sys

def extract_function_content(content, skip_includes=True):
    """Extract function content, optionally skipping #include lines."""
    lines = content.split('\n')
    result_lines = []
    
    for line in lines:
        # Skip #include lines if requested
        if skip_includes and re.match(r'^\s*#include\s', line):
            continue
        # Keep #undef lines
        if re.match(r'^\s*#undef\s', line):
            result_lines.append(line)
            continue
        # Keep everything else
        result_lines.append(line)
    
    # Remove leading/trailing empty lines
    while result_lines and not result_lines[0].strip():
        result_lines.pop(0)
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()
    
    return '\n'.join(result_lines)

# Process bt-serial files
bt_dir = 'bt-serial'
bt_main = os.path.join(bt_dir, 'bt.c')

# Read main bt.c
try:
    with open(bt_main, 'r') as f:
        bt_content = f.read()
except Exception as e:
    print(f"Error reading {bt_main}: {e}")
    sys.exit(1)

# Add math.h include after the existing includes (needed by error.c)
# Check if math.h is already included
if '#include <math.h>' not in bt_content and '#include "math.h"' not in bt_content:
    # Insert after the last #include
    lines = bt_content.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        # After the last #include, add math.h
        if line.strip().startswith('#include'):
            # Check if next line is not an include
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith('#include'):
                new_lines.append('#include <math.h>')
    bt_content = '\n'.join(new_lines)

# Append other files
bt_files = [
    ('add.c', True),
    ('error.c', True),
    ('exact_rhs.c', True),
    ('initialize.c', True),
    ('rhs.c', False),  # Keep #undef from rhs.c
    ('x_solve.c', True),
    ('y_solve.c', True),
    ('z_solve.c', True),
]

for filename, skip_includes in bt_files:
    filepath = os.path.join(bt_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()
    func_content = extract_function_content(content, skip_includes=skip_includes)
    bt_content += '\n\n' + func_content

# Write concatenated bt.c
with open(bt_main, 'w') as f:
    f.write(bt_content)

print(f"Concatenated bt.c written to {bt_main}")

# Process lu-serial files
lu_dir = 'lu-serial'
lu_main = os.path.join(lu_dir, 'lu.c')

# Read main lu.c
with open(lu_main, 'r') as f:
    lu_content = f.read()

# Append other files
lu_files = [
    ('blts.c', True),
    ('buts.c', True),
    ('erhs.c', True),
    ('jacld.c', True),
    ('jacu.c', True),
    ('l2norm.c', True),
    ('rhs.c', True),
    ('ssor.c', True),
]

for filename, skip_includes in lu_files:
    filepath = os.path.join(lu_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()
    func_content = extract_function_content(content, skip_includes=skip_includes)
    lu_content += '\n\n' + func_content

# Write concatenated lu.c
with open(lu_main, 'w') as f:
    f.write(lu_content)

print(f"Concatenated lu.c written to {lu_main}")

#!/bin/bash
set -e

# Function to extract content without #include lines (but keep #undef)
extract_content() {
    local file="$1"
    local keep_undef="$2"
    sed -E '/^\s*#include\s/d' "$file" | sed -n '/^[[:space:]]*#undef/p; /^[^#]/,$p'
}

# Process bt-serial
cd bt-serial

# Start with bt.c and add math.h if needed
if ! grep -q '#include.*math\.h' bt.c; then
    sed -i '/#include "print_results.h"/a #include <math.h>' bt.c
fi

# Append other files (skip #include, keep #undef for rhs.c)
for file in add.c error.c exact_rhs.c initialize.c; do
    extract_content "$file" >> bt.c
    echo "" >> bt.c
done

# For rhs.c, keep #undef lines
sed -E '/^\s*#include\s/d' rhs.c | sed -n '/^[[:space:]]*#undef/p; /^[^#]/,$p' >> bt.c
echo "" >> bt.c

# Append remaining files
for file in x_solve.c y_solve.c z_solve.c; do
    extract_content "$file" >> bt.c
    echo "" >> bt.c
done

cd ..

# Process lu-serial
cd lu-serial

# Start with lu.c
# Append other files
for file in blts.c buts.c erhs.c jacld.c jacu.c l2norm.c rhs.c ssor.c; do
    extract_content "$file" >> lu.c
    echo "" >> lu.c
done

cd ..

echo "Concatenation complete!"

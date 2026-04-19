**Goal:** Your sole purpose is to ensure the candidate code at `{kernel_dir}` (file(s): {file_list_str}) is numerically identical to the golden reference at `{golden_file_str}`. You will achieve this by instrumenting both with `gate.h` macros and fixing any discrepancies in the candidate code.

**MANDATORY:** You must not fallback to CPU implementation. The code must run on the GPU. If correctness cannot be achieved explain why in details.

**You must** {instructions_context}

**Context:**
- You are activated **after** an optimization step has modified the file(s): {file_list_str} in `{kernel_dir}`.
{backup_msg}

- gate macros are located in `{abs_gate_sdk}/gate.h`.

---

### Your Task (Step-by-Step Workflow)

1.  **Instrument Golden Reference (if needed):**
    * Ensure `{golden_file_str}` includes `#include "gate.h"`.
    * Gate Macros should only be included in code files and not header files.
    * After the main computation, add `GATE_CHECKSUM_*` or `GATE_STATS_*` macros to the golden file to capture the final state of all primary result buffers. This is your "source of truth". *You should only need to do this once.*

2.  **Instrument Candidate Code:**
    * Ensure the candidate file(s) in `{kernel_dir}` include `#include "gate.h"`:
{file_listing}
    * Add the **exact same GATE macros** as the golden reference, observing the same variables. The metric names, data types, and sample counts (`n` for stats) must match perfectly.

3.  **Build and Run Check:**
    * From the `{kernel_dir}` directory, run the following commands in order:
        1.  `{clean_cmd}`
        2.  `make -f Makefile.nvc check-correctness > supervisor_output.txt 2>&1`  
    
    * The output of the check-correctness command will be saved to `supervisor_output.txt` in `{kernel_dir}/` for inspection.

4.  **Debug and Fix (Iterate if Needed):**
    * **If the check passes:** Your job is done. Stop and output the final, correct code in the file(s) in `{kernel_dir}`.
    * **If the check fails:**
        a. Read `supervisor_output.txt` to analyze the failure output from the GATE check.
        b. Make the **absolute minimum change** to the relevant file(s) to fix the numerical error.
        c. Loop back to Step 3 (Build and Run Check). **Do not stop until the check passes.**

---

### Debugging Strategy

When a check fails, use this hierarchy of likely causes:

{debugging_strategy}

---

### Strict Rules

* **BEFORE** any time you you want to compile the code, you must run `{clean_cmd}` in `{kernel_dir}`.
* **DO NOT** perform any performance optimizations. Your only goal is correctness.
* **DO NOT** modify Makefiles, input data, or build commands.
* **DO NOT** change the golden reference file (`{golden_file_str}`) except to add `gate.h` and GATE macros.
* **ONLY** edit the candidate file(s) in `{kernel_dir}`:
{file_listing}
{strict_rules_extra}

**Never / Forbidden**
- Run commands that read / write to files outside of your current working directory.
- NO GIT COMMANDS
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY

**Output Files:**
- `supervisor_output.txt` - Contains the output of `make -f Makefile.nvc check-correctness` command
- This file will be created in `{kernel_dir}/` and can be inspected to see the GATE check results

**Deliverable:**
- The final, corrected source code for the file(s) in `{kernel_dir}` that successfully passes the `make -f Makefile.nvc check-correctness`:
{file_listing}
- The `supervisor_output.txt` file containing the check-correctness output

#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- 1. Create a temporary Python script for smart comparison ---
cat << 'EOF' > temp_compare.py
import json
import sys

# List of all possible keys that represent the "cost" of the path (Phase 1)
COST_KEYS = {
    "minimum_time", 
    "minimum_distance", 
    "minimum_time/minimum_distance", 
    "minumum_time/minimum_distance"
}

def get_cost_value(obj):
    """Helper to find the cost value regardless of which key is used."""
    for key in COST_KEYS:
        if key in obj:
            return obj[key]
    return None

def compare_json(file1, file2):
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            d1 = json.load(f1)
            d2 = json.load(f2)
    except Exception as e:
        print(f"Error loading JSON files: {e}")
        return "Error"

    results1 = d1.get("results", [])
    results2 = d2.get("results", [])

    if len(results1) != len(results2):
        print(f"Length mismatch! Your output has {len(results1)} results, expected {len(results2)}")
        return "Error"

    # Iterate through each result object
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        id_val = r1.get("id", "Unknown")
        
        # --- Check 1: Compare Phase 1 Costs (Smart Match with Tolerance) ---
        val1 = get_cost_value(r1)
        val2 = get_cost_value(r2)

        if val1 is not None and val2 is not None:
            try:
                f1_val = float(val1)
                f2_val = float(val2)
                if abs(f1_val - f2_val) > 1.0:
                    print(f"  [ID {id_val}] Cost mismatch")
                    print(f"      Yours: {f1_val}")
                    print(f"      Expct: {f2_val}")
            except ValueError:
                 print(f"  [ID {id_val}] Cost is not a number")
        elif (val1 is None) != (val2 is None):
             if r1.get("possible", True) and r2.get("possible", True):
                 print(f"  [ID {id_val}] Missing cost key in one file")

        # --- Check 2: Compare All Keys ---
        for key in r2:
            if key in COST_KEYS: continue # Already handled above
            if key == "processing_time": continue # Ignore

            val_exp = r2[key]
            val_user = r1.get(key)

            # --- PHASE 2 SPECIFIC CHECKS ---
            
            # Case A: K-Shortest Paths ("paths" array)
            if key == "paths":
                if not isinstance(val_user, list) or len(val_user) != len(val_exp):
                    print(f"  [ID {id_val}] 'paths' list mismatch (count or type)")
                    continue
                
                for idx, (u_path_obj, e_path_obj) in enumerate(zip(val_user, val_exp)):
                    # Check Length (+/- 1.0)
                    u_len = u_path_obj.get("length")
                    e_len = e_path_obj.get("length")
                    if u_len is not None and e_len is not None:
                        if abs(float(u_len) - float(e_len)) > 1.0:
                            print(f"  [ID {id_val}] Path {idx} length mismatch")
                            print(f"      Yours: {u_len}")
                            print(f"      Expct: {e_len}")
                    
                    # Check Node Sequence (Exact)
                    if u_path_obj.get("path") != e_path_obj.get("path"):
                        print(f"  [ID {id_val}] Path {idx} node sequence mismatch")

            # Case B: Approx Shortest Path ("distances" array)
            elif key == "distances":
                if not isinstance(val_user, list) or len(val_user) != len(val_exp):
                    print(f"  [ID {id_val}] 'distances' list mismatch")
                    continue

                for idx, (u_dist_obj, e_dist_obj) in enumerate(zip(val_user, val_exp)):
                    # Check Approx Distance (+/- 10.0)
                    u_dist = u_dist_obj.get("approx_shortest_distance")
                    e_dist = e_dist_obj.get("approx_shortest_distance")
                    if u_dist is not None and e_dist is not None:
                        if abs(float(u_dist) - float(e_dist)) > 10.0:
                            print(f"  [ID {id_val}] Approx dist {idx} mismatch")
                            print(f"      Yours: {u_dist}")
                            print(f"      Expct: {e_dist}")

                    # Check Source/Target (Exact)
                    if u_dist_obj.get("source") != e_dist_obj.get("source") or \
                       u_dist_obj.get("target") != e_dist_obj.get("target"):
                        print(f"  [ID {id_val}] Approx dist {idx} source/target mismatch")

            # --- PHASE 1 CHECKS ---
            elif key == "path":
                # Exact list match for single shortest path
                if val_user != val_exp:
                    print(f"  [ID {id_val}] Path mismatch")
                    print(f"      Yours: {val_user}")
                    print(f"      Expct: {val_exp}")
            
            # --- GENERIC CHECK ---
            else:
                # Generic match (id, possible, done, etc)
                if val_user != val_exp:
                    print(f"  [ID {id_val}] Mismatch for '{key}'")
                    print(f"      Yours: {val_user}")
                    print(f"      Expct: {val_exp}")

if __name__ == "__main__":
    compare_json(sys.argv[1], sys.argv[2])
EOF

# --- 2. Start the Test Batch ---

echo -e "${CYAN}Starting Test Batch...${NC}\n"

# Loop from 0 to 9
for i in {0..3}
do
    # Define file paths
    GRAPH="./../tests/testcase$i/graph.json"
    QUERIES="./../tests/testcase$i/queries2.json"
    
    OUT_DIR="./../myresults/testcase$i"
    MY_OUTPUT="$OUT_DIR/output2.json"
    
    # Assuming output.json
    EXPECTED_OUTPUT="./../results/testcase$i/output2.json"

    # Create directory if it doesn't exist
    if [ ! -d "$OUT_DIR" ]; then
        mkdir -p "$OUT_DIR"
    fi

    # If file exists, remove it (wipe clean). Then create a fresh empty file.
    if [ -f "$MY_OUTPUT" ]; then
        rm "$MY_OUTPUT"
    fi
    touch "$MY_OUTPUT"


    # Execute the binary
    ./phase2 "$GRAPH" "$QUERIES" "$MY_OUTPUT" > /dev/null

    # Compare outputs
    echo -e "Comparing Test Case $i..."
    
    if [ -f "$EXPECTED_OUTPUT" ]; then
        # Run the python comparator
        DIFFERENCES=$(python3 temp_compare.py "$MY_OUTPUT" "$EXPECTED_OUTPUT")
        
        if [ -z "$DIFFERENCES" ]; then
            echo -e "${GREEN}Test Case $i Passed!${NC}"
        else
            echo -e "${RED}Differences found in Test Case $i:${NC}"
            echo "$DIFFERENCES"
        fi
    else
        echo -e "${YELLOW}Warning: Expected result file not found at $EXPECTED_OUTPUT${NC}"
    fi

    echo -e "${CYAN}=== End of Test Case $i ===${NC}\n"

done

# Clean up
rm temp_compare.py
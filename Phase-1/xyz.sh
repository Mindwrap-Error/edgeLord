#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}Starting Test Batch...${NC}\n"

# Loop from 0 to 9
for i in {0..1}
do
    # Define file paths
    GRAPH="./../tests/testcase$i/graph.json"
    QUERIES="./../tests/testcase$i/queries1.json"
    
    OUT_DIR="./../myouts/out$i"
    MY_OUTPUT="$OUT_DIR/output.json"
    
    EXPECTED_OUTPUT="./../results/testcase$i/output1.json"

    # 1. Create directory if it doesn't exist
    if [ ! -d "$OUT_DIR" ]; then
        mkdir -p "$OUT_DIR"
    fi

    # 2. Create output file if it doesn't exist (though your C++ code should likely create/overwrite it)
    if [ ! -f "$MY_OUTPUT" ]; then
        touch "$MY_OUTPUT"
    fi

    # 3. Execute the binary
    # We suppress stdout to keep the terminal clean for the diffs, 
    # remove "> /dev/null" if you want to see your program's prints.
    ./a.out "$GRAPH" "$QUERIES" "$MY_OUTPUT"

    # 4. Compare outputs
    echo -e "Comparing Test Case $i..."
    
    if [ -f "$EXPECTED_OUTPUT" ]; then
        # Run diff and print it
        diff "$MY_OUTPUT" "$EXPECTED_OUTPUT"
    else
        echo -e "${RED}Warning: Expected result file not found at $EXPECTED_OUTPUT${NC}"
    fi

    # 5. Print colored end-of-case message
    echo -e "${GREEN}=== End of differences for Test Case $i ===${NC}\n"

done
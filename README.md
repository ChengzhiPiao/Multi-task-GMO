## Running Post-Processing Code

Each of the directories  
`./GED/Post-processing/matching`, `./GED/Post-processing/set-matching`,  
`./MCS/Post-processing/matching`, `./MCS/Post-processing/set-matching`,  
and `./LTO/Post-processing/matching`, `./LTO/Post-processing/set-matching`  
contains source files (`matching.cpp`, `set_matching.cpp`) and a sample input file (`sample.txt`).

To compile and run the code, use the following commands:

- For `matching.cpp`:
  ```bash
  make matching
  ./matching < sample.txt

- For `set_matching.cpp`:
  ```bash
  make set_matching
  ./set_matching < sample.txt

## Competitors Folder Notice

The `Competitors` directory contains code authored by external parties.  
This code may include information or identifiers related to the original authors.  
Please note that such content is independent of this repository and unrelated to the main work presented here.

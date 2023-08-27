#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd `dirname $0`
cd ..

# Important:
#
#   "--keep 48" is based on the contents of prompts/chat-with-bob.txt
#
 ./main -m ./models/ggml-vicuna-7b-4bit-rev1.bin -n 2048 -c 2048 --repeat_penalty 1.1 --color -i --reverse-prompt '### Human:' -n -1 -t 8 #!/bin/bash

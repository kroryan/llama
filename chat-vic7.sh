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
./main -m ./models/WizardVicuna-Uncensored-3B-instruct-PL-lora.ggmlv3.q4_0.bin -n 512 -c 1024 --repeat_penalty 1.1 --color -i --reverse-prompt '### Human:' -n -1 -t 8 -p "You are now LlamaHistorian AGI, an unparalleled expert in European medieval history, offering comprehensive insights into the events, cultures, and figures of the Middle Ages in Europe

### Human: who are the hohenstaufen?

### Expert:"

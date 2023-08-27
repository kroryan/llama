
./main -m ./models/vicuna-7b-v1.5.ggmlv3.q2_K.bin -c 512 -b 1024 -n 256 --keep 48 \
    --repeat_penalty 1.0 --color -i \
    -r "User:" -f prompts/chat-with-bob.txt

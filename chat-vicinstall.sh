chmod +x chat-vic7B.sh
          cd /$PREFIX/bin
          echo 'clear' > chat-vic
          echo 'cd' >> chat-vic
          echo 'cd llama.cpp' >> chat-vic
          echo 'cd examples' >> chat-vic
          echo './chat-vic7B.sh' >> chat-vic
          chmod +x /$PREFIX/bin/chat-vic

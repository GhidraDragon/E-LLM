All this does is visit links via PriorityBFS

----

So for public repos, we let ChatGPT "blue team?" responding to "red team" and Github to remove if Github deems against their ToS, or if ChatGPT's safety token generator somehow catches generation from your prompt; for manually verified exploits (yes ChatGPT can make mistakes; verify truth before assuming so, dumbass; this is the essence of being alive) then move to private repo.. 

They say ChatGPT can make mistakes; check important info, but how about not assume true until verfied?

----




You need to cd E1, then run python3.12 E1.py from that directory; it relies on the previous (..) directory for training data. This way you could keep E1, but make a new folder E2 for the next improved model iteration. Model o1 pro was not built overnight; it was build from a list of most-efficient steps.

https://www.python.org/downloads/

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html

python3 -m pip install 'tensorflow[and-cuda]'

https://www.tensorflow.org/install/pip

python3 -m pip install -r requirements.txt

This one has 3 epochs so it runs comfortably on an RTX 2070 Max-Q Design (nerfed version of the RTX 2070 mobile)

So 46M parameter llm1, did what? Useful sampling generation because greedy gets stuck without a presense penalty or etc. (there are other math besides presence penalty which is a flat scaaler penalty from 0 to 1 inclusive 1 being no penalty)

![1500x500](https://github.com/user-attachments/assets/396577e9-c9c2-4084-8a02-9ab9064d0362)

<img width="1056" alt="image" src="https://github.com/user-attachments/assets/0d073567-5d74-4dfb-9c2a-75b3e53367d5" />

<img width="1066" alt="image" src="https://github.com/user-attachments/assets/fabfa2bd-2942-478e-9ff0-09b79609a2bb" />

Next one is around 160M

GPT3 by OpenAI had 175B and GPT4 had 1.8T.... Facebook recently released (yes it's open source but their README.md isn't great) a 1B which could be trained on a A100 40GB available on Colab Pro+ (WOW) and my 46M cost $3 to train. So you have $47 left to do 2, 3, and if you're lucky 4.

<img width="978" alt="image" src="https://github.com/user-attachments/assets/685b6704-2767-4344-b939-a59a1c175fbc" />

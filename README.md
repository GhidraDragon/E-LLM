You need to cd E1, then run python3.12 E1.py from that directory; it relies on the previous (..) directory for training data. This way you could keep E1, but make a new folder E2 for the next improved model iteration. Model o1 pro was not built overnight; it was build from a list of most-efficient steps.

https://www.python.org/downloads/

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html

python3 -m pip install 'tensorflow[and-cuda]'

https://www.tensorflow.org/install/pip

python3 -m pip install -r requirements.txt

This one has 3 epochs so it runs comfortably on an RTX 2070 Max-Q Design (nerfed version of the RTX 2070 mobile)

![20250318_101626](https://github.com/user-attachments/assets/bb72818d-1574-4b3a-a5c9-4a403c5f6948)

If you're missing the MITRE CVE


The one runnig on my RTX 4070 mobile has 10 epochs, and other upgrades too.

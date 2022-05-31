# PokeMeow With PyTorch

A farm bot for Pokemeow. Solves captcha with custom trained weights with YOLOv5 and PyTorch.

### Prerequisites :
- Install latest version of Python https://www.python.org/downloads/
- If running on Ubuntu "sudo apt install libgl1-mesa-glx"
- Install Git https://git-scm.com/downloads
- Install 2Captcha "pip install 2captcha-python"

### Installation with terminal :

```
# clone this repo
git clone https://github.com/jakelovescoding/PokeTorch.git

# change directory
cd PokeTorch

# install requirements
pip install -r requirements.txt

```

### Edit config.py

Authorization:

Find auth token by watching this video:

https://www.youtube.com/watch?v=YEgFvgg7ZPI

Channel Id is the last section of this url: (in this case 954919398404141076):
<img align="center" src="readmepic/channel_id.png" width="1000">


```
captcha = None # 2captcha api token for backup (None if not using 2captcha)
channel = "" # channel for farming
auth = "" # authorization token
```

### Run:

To start farming, in terminal type:

```
python3 main.py
```





# gan-demo
Simple GAN Demo with Anime picture in 508

### Env Setup
先進508, run

    sh pre_run.sh
    
    hrun -N l12 bash main.sh

### Remote Setting
In local machine, run
    
    ssh -N -L localhost:8889:l12:8889 <user@remote.ip>

Finally, use the following url,或是直接複製jupyter給你的網址
    
    localhost:8889/<token>

### Run in jupyter
    %run dcgan.py
    %run wgan.py


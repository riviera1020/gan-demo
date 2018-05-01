# gan-demo
Simple GAN Demo with Anime picture

### Env Setup
先進508, run
    
    hrun -N l04 -C bash pre_run.sh

### Remote Setting
In remote machine(508), run

    hrun -N l04 jupyter notebook --no-browser --port=8889 --allow-root --ip=0.0.0.0

In local machine, run
    
    ssh -N -L localhost:8889:l04:8889 <user@remote.ip>

Finally, use the following url
    
    localhost:8889/<token>

### Run
    %run dcgan.py
    %run wgan.py


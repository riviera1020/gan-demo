# gan-demo
Simple GAN Demo with Anime picture

### Remote Setting
In remote machine(battle ship), run
    jupyter notebook --no-browser --port=8889
In local machine, run
    ssh -N -L localhost:8888:localhost:8889 <user@remote.ip>
Finally, use the following url
    localhost:8888/

### Run
    python dcgan.py --inputs_dir <inputs_dir>
    python wgan.py --inputs_dir <inputs_dir>

### Data
    /nfs/Valkyrie/riviera1020/extra_data/one_tag_images/

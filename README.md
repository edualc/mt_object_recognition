# mt_object_recognition

This is the repository for the code used in my master thesis, titled "Leveraging Neuroscience for Deep Learning Based Object Recognition".


### Installation

Run the makefile with ```make init```. After that the environment will be created and necessary dependencies are installed.

The code is based on Python version ```3.8.10```.

When you later just want to run the environment, use the ```scripts/activate_env.sh``` shell file. For using Jupyter, after activating the environment run ```scrips/start_jupyter.sh```.

The Optuna hyperparameter tuning library uses SQLite for synchronization of parallel trial runs, if you intend to use it you might have to install SQLite on your machine: ```sudo apt-get install sqlite```.
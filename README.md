# Open-TI: Open Traffic Intelligence with Augmented Language Model

## Introduction

Open-TI is a revolutionary traffic intelligence model that bridges the industry-academic gap in intelligent transportation. It leverages large language models to execute complex traffic analysis tasks, making it the first to seamlessly integrate external packages based on conversations. Beyond analysis, Open-TI can train traffic signal control policies, explore demand optimizations, and communicate with control agents like ChatZero for efficient task execution. With a formal structure and open-ended design, Open-TI invites community-driven enhancements, emphasizing its pivotal role in advancing intelligent transportation systems.


## The overview of the Open-TI functionalities
![overview](./assets/Overview.png)

## The design framework of Open-TI
![framework](./assets/frameworkdesign.png)

## The Open-TI conversation interface
![interface](./assets/interface.png)


# Installation

## Source

Our code is based on Python version 3.9 and Pytorch version 1.11.0.
Open-TI provides installation from the source code. 
Please execute the following command to install and configure our environment.


```Powershell
clone https://github.com/DaRL-LibSignal/OpenTI.git
```


## Simulator environment configuration
<br />
Though CityFlow, SUMO,  and LibSignal are stable under Windows and Linux systems, we still recommend users work under the Linux system.<br><br>

### CityFlow Environment
<br />

To install CityFlow simulator, please follow the instructions on [CityFlow Doc](https://cityflow.readthedocs.io/en/latest/install.html#)


```
sudo apt update && sudo apt install -y build-essential cmake

git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
pip install .
```
To test configuration:
```
import cityflow
env = cityflow.Engine
```
<br>

### SUMO Environment
<br />

To install SUMO environment, please follow the instructions on [SUMO Doc](https://epics-sumo.sourceforge.io/sumo-install.html#)

```
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig

git clone --recursive https://github.com/eclipse/sumo

export SUMO_HOME="$PWD/sumo"
mkdir sumo/build/cmake-build && cd sumo/build/cmake-build
cmake ../..
make -j$(nproc)
```
To test installation:
```
cd ~/DaRL/sumo/bin
./sumo
```

To add SUMO and traci model into the system PATH, execute the code below:
```
export SUMO_HOME=~/DaRL/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```
To test configuration:
```
import libsumo
import traci
```
<br>

### LibSignal Environment
<br />

To install LibSignal environment, please follow the instructions on [LibSignal Doc](https://darl-libsignal.github.io/#download)

```
git clone https://github.com/DaRL-LibSignal/LibSignal.git
cd LibSignal
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

```

<br>


## Configuration

### LLM Configuration

First, you need to configure OpenAI-Key. Please create a `./config.yaml` file in the root directory and add the following content to the file (please modify the content to your own settings):

```yaml
OPENAI_API_TYPE: 'openai' #'azure'  OR 'openai'
# for 'openai'
OPENAI_KEY: '' # your openai key
```

Here we recommend using ChatGPT-3.5 to run as LLM. If you want to use your own LLM, please refer to [LangChain-Large Language Models](https://python.langchain.com/docs/modules/model_io/models/) to define Your own LLM. In this case, please modify the following sections in `./DataProcessBot.py` and `./SimulationProcessBot.py` to configure your own LLM.


## Start Running
<br />

If you install all the requirments below, you can run Open-TI now.

```
cd Open-TI
python3 executor.py
```

<br>

## Citation

```
@misc{,
      title={Open-TI: Open Traffic Intelligence with Augmented Language Model}, 
      author={Longchao Da and Kuanru Liou and Tiejin Chen and Xuesong Zhou and Xiangyong Luo and Yezhou Yang and Hua Wei},
      year={2023},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

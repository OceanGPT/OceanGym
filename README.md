<h1 align="center"> 🌊 OceanGym 🦾 </h1>
<h3 align="center"> A Benchmark Environment for Underwater Embodied Agents </h3>

<p align="center">
  🌐 <a href="https://123" target="_blank">Home Page</a>
  📄 <a href="https://arxiv.org/abs/123" target="_blank">ArXiv Paper</a>
  🤗 <a href="https://huggingface.co/123" target="_blank">Hugging Face</a>
  🌈 <a href="https://123" target="_blank">Google Drive</a>
  ☁️ <a href="https://123" target="_blank">Baidu Drive</a>
</p>

- [⚙️ Set up Environment](#️-set-up-environment)
  - [Deploy HoloOcean](#deploy-holoocean)
  - [Cloning](#cloning)
  - [Opening \& Prepping Project](#opening--prepping-project)
  - [Setting up VSCode](#setting-up-vscode)
  - [Compiling](#compiling)
  - [Launching Game Live](#launching-game-live)
- [🧠 Decision Task](#-decision-task)
  - [Quick Start](#quick-start)
  - [Target Object Locations](#target-object-locations)
  - [Evaluation Criteria](#evaluation-criteria)
- [👀 Perception Task](#-perception-task)
  - [Using the Bench to Eval](#using-the-bench-to-eval)
  - [Collecting Image Data](#collecting-image-data-optional)
- [🌻 Acknowledgement](#-acknowledgement)
- [🚩 Citation](#-citation)

# ⚙️ Set up Environment

This project is based on the `HoloOcean` environment.

```bash
conda create -n oceangym python=3.13.2
conda activate oceangym
```

After that, make sure the `HoloOcean` is ready:

## Deploy HoloOcean

Refence at: https://byu-holoocean.github.io/holoocean-docs/v2.0.1/develop/start.html#getting-started

The steps in this page are only necessary if you want to change HoloOcean’s C++ code or if you want to use Unreal Engine to develop new worlds. It is not necessary for making changes to the Python API or for using HoloOcean as a user.

While developing for Unreal Engine is supported on both Linux and Windows, we have found the Unreal Engine development tools to be more stable and straightforward in Windows.

Developing in HoloOcean requires the following additional dependencies:

- [Unreal Engine 5.3](https://dev.epicgames.com/documentation/en-us/unreal-engine/installing-unreal-engine?application_version=5.3)
- [Visual Studio 2022 or 2019](https://visualstudio.microsoft.com/)
  - See UE [documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-visual-studio-development-environment-for-cplusplus-projects-in-unreal-engine?application_version=5.3) for setup
- git
- Python installation environment (we recommend conda)
- numpy version >= 2.0.0

HoloOcean 2.0 requires Numpy version 2. This is a breaking change from HoloOcean 1.0, which used numpy version 1. Please ensure you are using the version of numpy appropriate for the version of HoloOcean you are using.

## Cloning

Refence at: https://byu-holoocean.github.io/holoocean-docs/v2.0.1/develop/start.html#cloning

For running holoocean live, you’ll need to setup both the C++ and Python portions of HoloOcean.

> - Clone [holoocean](https://github.com/byu-holoocean/HoloOcean).
> - Navigate into the local repository.
> - Checkout to the branch you want to develop on, likely the develop branch, which you can access through `git checkout develop`,
>   - Alternatively, create your own new branch for the feature or addition through `git checkout -b [your branch name] [the branch you want to branch off of]`

You can now install the Python package by running `pip install -e client/` (or `pip install .`). Make sure not to skip this step - reinstalling the package is necessary to apply the changes from switching branches.

## Opening & Prepping Project

Refence at: https://byu-holoocean.github.io/holoocean-docs/v2.0.1/develop/start.html#opening-prepping-project

To open the HoloOcean project in the Unreal Editor, find the `holodeck.uproject` file in the `holoocean/engine` directory. Double-click this file, and choose “5.3” if an engine version dialog opens up. Alternatively, open the Unreal Editor and select “Open Project” from the main menu.

If you get a dialog that says “The following modules are missing or built with a different engine version”, click “yes” to rebuild the project. This will take a few minutes, and you may get a few errors. If you do, click “yes” to all of them.

![../_images/the-build-error.png](https://byu-holoocean.github.io/holoocean-docs/v2.0.1/_images/the-build-error.png)

If you continue to get errors, please reference [Troubleshooting](https://byu-holoocean.github.io/holoocean-docs/v2.0.1/develop/troubleshooting.html#troubleshooting).

In the Unreal Editor, go to Platforms -> <your operating system> -> Cook Content. After a few minutes you should get a success popup in the lower right.

## Setting up VSCode

Refence at: https://byu-holoocean.github.io/holoocean-docs/v2.0.1/develop/start.html#setting-up-vscode

If you would like to use VSCode instead of Visual Studio for HoloOcean, you can do the following:

- In Unreal Editor, Go to Edit -> Editor Preferences
- Then go to General -> Source Code -> Source Code Editor and select Visual Studio Code
- Once this is done you should now be able to generate a new Visual Studio Code project using File -> Generate Visual Studio Code Project
- To open up Visual Studio Code go to File -> Open Visual Studio Code

## Compiling

Refence at: https://byu-holoocean.github.io/holoocean-docs/v2.0.1/develop/start.html#compiling

You need to recompile the project after making any changes to the C++ code.

Generally you can compile by clicking the “Compile” button in Unreal Editor at the bottom-right corner of the screen. It might be hidden if your screen isn’t wide enough. To ensure it compiles properly, click on the 3 vertical dots next to the compile button and make sure “Enable Live Coding” is turned off.

![../_images/compile-button-location.png](https://byu-holoocean.github.io/holoocean-docs/v2.0.1/_images/compile-button-location.png)

If you want to use Visual Studio to compile HoloOcean, you will need to generate the .sln file by right clicking on the `holodeck.uproject` file within the engine folder, and then clicking on “Generate Visual Studio Project Files”. If you are using Windows 11, after right clicking on the uproject file, you will need to click “Show more options” in order to find the right option.

![../_images/generate-vs-files.png](https://byu-holoocean.github.io/holoocean-docs/v2.0.1/_images/generate-vs-files.png)

Double click the generated .sln file to open it within Visual Studio. From there, you can right click on the Solution on the right hand side of the screen and select “Build Solution”.

![../_images/build-in-visual-studio.png](https://byu-holoocean.github.io/holoocean-docs/v2.0.1/_images/build-in-visual-studio.png)

## Launching Game Live

Refence at: https://byu-holoocean.github.io/holoocean-docs/v2.0.1/develop/start.html#launching-game-live

To avoid having to package the project anytime you want to see changes to your code, you can play the game live from Unreal Editor and then attached your Python code to it. This is referred to as “running in standalone”. This is a multi-step process, as follows.

If developing a sonar module, in UE5 click the 3 dots next to the “Play” button in the top toolbar, and click “Advanced Settings”. Add the following line to “Additional Launch Parameters”

![../_images/standalone-game.png](https://byu-holoocean.github.io/holoocean-docs/v2.0.1/_images/standalone-game.png)

These are all in meters. Tweak them as needed.

In addition, the -log parameter is useful for being able to close the game window easily, as well as for seeing log messages for debugging purposes.

# 🧠 Decision Task

> All commands are applicable to **Windows** only, because it requires full support from the `UE5 Engine`.

## Quick Start

1. **Install Dependencies**

   Make sure you have Python 3.10+ and the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Install HoloOcean Ocean World**

   In a Python shell or command line, run:

   ```python
   import holoocean
   holoocean.install("Ocean")
   ```

   Or:

   ```bash
   python -c "import holoocean; holoocean.install('Ocean')"
   ```

3. **Add World Files**

   Place `OceanGym.json` and `OceanGym_sonar.json` into:

   ```
   C:\Users\Windows\AppData\Local\holoocean\2.0.0\worlds\Ocean
   ```

4. **Configure the Environment**

   Edit `config.yaml` in  "\navigation\config.yaml" or select an existing config file. Set `base_path`, scenario parameters, etc.

5. **Open the World**

   1. First, open the project

  <img src="asset\img\a1.png" alt="method" style="zoom: 50%;" />

   2. Find the **HAIDI** map in **demo** directory

  <img src="asset\img\a2.jpg" alt="method" style="zoom: 50%;" />

   1. Run the project

  <img src="asset\img\a3.png" alt="method" style="zoom: 50%;" />

1. **Run a Task Script**

   For example, to run task 4:

   ```bash
   python task\navigation\tasks\task4.py
   ```

   Follow the keyboard instructions or switch to LLM mode for automatic decision-making.


   **Keyboard Control Guide:**

   | Key         | Action                        |
   |-------------|------------------------------|
   | W           | Move Forward                 |
   | S           | Move Backward                |
   | A           | Move Left                    |
   | D           | Move Right                   |
   | J           | Turn Left                    |
   | L           | Turn Right                   |
   | I           | Move Up                      |
   | K           | Move Down                    |
   | M           | Switch to LLM Mode           |
   | Q           | Exit                         |

   > You can use WASD for movement, J/L for turning, I/K for up/down.
   > Press `M` to switch to large language model mode (may cause temporary lag).
   > Press `Q` to exit.
7. **View Results**

   Logs and memory files are automatically saved in the `log/` and `memory/` directories.

8. **Evaluate the results**

   Place the generated `memory` and `important_memory` files into the corresponding `point` folders.
   Then, set the evaluation paths in the `evaluate.py` file.

   We provide 6 experimental evaluation paths. In `evaluate.py`, you can configure them as follows:

   ```python
   eval_roots = [
       os.path.join(eval_root, "main", "gpt4omini"),
       os.path.join(eval_root, "main", "gemini"),
       os.path.join(eval_root, "main", "qwen"),
       os.path.join(eval_root, "migration", "gpt4o"),
       os.path.join(eval_root, "migration", "qwen"),
       os.path.join(eval_root, "scale", "qwen"),
   ]
   ```

   To run the evaluation:

   ```bash
   python decision\utils\evaluate.py
   ```

   The generated results will be saved under the `\eval\navigation` folder.

## Target Object Locations

We have provided eight tasks. For specific task descriptions, please refer to the paper.

The following are the coordinates for each target object in the environment (in meters):

- **MINING ROBOT**:
  (-71, 149, -61), (325, -47, -83)
- **OIL PIPELINE**:
  (345, -165, -32), (539, -233, -42), (207, -30, -66)
- **OIL DRUM**:
  (447, -203, -98)
- **SUNKEN SHIP**:
  (429, -151, -69), (78, -11, -47)
- **ELECTRICAL BOX**:
  (168, 168, -65)
- **WIND POWER STATION**:
  (207, -30, -66)
- **AIRCRAFT WRECKAGE**:
  (40, -9, -54), (296, 78, -70), (292, -186, -67)
- **H-MARKED LANDING PLATFORM**:
  (267, 33, -80)

---

## Evaluation Criteria

1. If the target is not found, use the final stopping position for evaluation.
2. If the target is found, use the closest distance to any target point.
3. For found targets:
   - Minimum distance ≤ 30: full score
   - 30 < distance < 100: score decreases proportionally
   - Distance ≥ 100: score is 0
4. Score composition:
   - One point: 100
   - Two points: 60 / 40
   - Three points: 60 / 20 / 20

# 👀 Perception Task

## Using the Bench to Eval

> All commands are applicable to **Linux**, so if you using **Windows**, you need to change the corresponding path representation (especially the slash).

### Import Data

First, you need download our data from [Hugging Face](TODO) or [Google Drive](TODO) or [Baidu Drive](TODO).

And then create a new `data` folder in the project root directory:

```bash
mkdir -p data/perception
```

Finally, put the downloaded data into the corresponding folder.

### Set your Model Parameters

Open a terminal in the root directory and set it directly, or use the script from [here](scripts/perception/eval.sh).

| parameter | function |
| ---| --- |
| model_template | The large language model message queue template you selected. |
| model_name_or_path | If it is an API model, it is the model name; if it is a local model, it is the path. |
| api_key | If it is an API model, enter your key. |
| base_url | If it is an API model, enter its baseful URL. |

```bash
MODELS_TEMPLATE="Yours"
MODEL_NAME_OR_PATH="Yours"
API_KEY="Yours"
BASE_URL="Yours"
```

### Simple Multi-views

All of these scripts evaluate the perception task, and the parameters are as follows:

| parameter | function |
| ---| --- |
| exp_name | Customize the name of the experiment to save the results. |
| exp_idx | Select the experiment number, or enter "all" to select all. |
| exp_json | JSON file containing the experiment label data. |
| images_dir | The folder where the experimental image data is stored. |

This command is used to evaluate the **multi-view** perception task under **high** illumination:

```bash
python perception/eval/mv.py \
    --exp_name Result_MV_highLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLight.json" \
    --images_dir "/data/perception/highLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

This command is used to evaluate the **context-based** perception task under **high** illumination:

```bash
python perception/eval/mv.py \
    --exp_name Result_MV_highLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLightContext.json" \
    --images_dir "/data/perception/highLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

This command is used to evaluate the **multi-view** perception task under **low** illumination:

```bash
python perception/eval/mv.py \
    --exp_name Result_MV_lowLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLight.json" \
    --images_dir "/data/perception/lowLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

This command is used to evaluate the **context-based** perception task under **low** illumination:

```bash
python perception/eval/mv.py \
    --exp_name Result_MV_lowLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLightContext.json" \
    --images_dir "/data/perception/lowLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

### Multi-views with Sonar

This command is used to evaluate the **multi-view** perception task under **high** illumination with **sonar** image:

```bash
python perception/eval/mvs.py \
    --exp_name Result_MVwS_highLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLight.json" \
    --images_dir "/data/perception/highLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

This command is used to evaluate the **context-based** perception task under **high** illumination with **sonar** image:

```bash
python perception/eval/mvs.py \
    --exp_name Result_MVwS_highLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLightContext.json" \
    --images_dir "/data/perception/highLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

This command is used to evaluate the **multi-view** perception task under **low** illumination with **sonar** image:

```bash
python perception/eval/mvs.py \
    --exp_name Result_MVwS_lowLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLight.json" \
    --images_dir "/data/perception/lowLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

This command is used to evaluate the **context-based** perception task under **low** illumination with **sonar** image:

```bash
python perception/eval/mvs.py \
    --exp_name Result_MVwS_lowLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLightContext.json" \
    --images_dir "/data/perception/lowLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

### Multi-views add Sonar Examples

This command is used to evaluate the **multi-view** perception task under **high** illumination with **sona** image **examples**:

```bash
python perception/eval/mvsex.py \
    --exp_name Result_MVwSss_highLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLight.json" \
    --images_dir "/data/perception/highLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

This command is used to evaluate the **context-based** perception task under **high** illumination with **sona** image **examples**:

```bash
python perception/eval/mvsex.py \
    --exp_name Result_MVwSss_highLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLightContext.json" \
    --images_dir "/data/perception/highLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

This command is used to evaluate the **multi-view** perception task under **low** illumination with **sona** image **examples**:

```bash
python perception/eval/mvsex.py \
    --exp_name Result_MVwSss_lowLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLight.json" \
    --images_dir "/data/perception/lowLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

This command is used to evaluate the **context-based** perception task under **low** illumination with **sona** image **examples**:

```bash
python perception/eval/mvsex.py \
    --exp_name Result_MVwSss_lowLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLightContext.json" \
    --images_dir "/data/perception/lowLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL
```

## Collecting Image Data (Optional)

### Modify Configuration File

The sample configuration files can be found in `asset/perception/map_config`. You need to copy this and paste it into your HoloOcean project's configuration.

### Collect Camera Images Only

This command is used to collect **camera** images only, and the parameters are as follows:

| parameter | function |
| ---| --- |
| scenario | The name of the json configuration file you want to replace. |
| task_name | Customize the name of the experiment to save the results. |
| rgbcamera | The camera directions you can choose. If select all, enter "all". |

```bash
python perception/task/init_map.py \
    --scenario without_sonar \
    --task_name "Exp_Camera_Only" \
    --rgbcamera "all"
```

### Collect Camera and Sonar Images

This command is used to collect both **camera** images and **sonar** images at same time:

```bash
python perception/task/init_map_with_sonar.py \
    --scenario with_sonar \
    --task_name "Exp_Add_Sonar" \
    --rgbcamera "FrontCamera"
```

# 🌻 Acknowledgement

~

# 🚩 Citation

If this paper or benchmark is helpful, please kindly cite as this, thanks!

```bibtex
@inproceedings{oceangym,
  title={OceanGym: A Benchmark Environment for Underwater Embodied Agents},
  ...
}
```

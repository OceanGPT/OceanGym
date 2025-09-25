<h1 align="center"> 🌊 OceanGym 🦾 </h1>
<h3 align="center"> A Benchmark Environment for Underwater Embodied Agents </h3>

<p align="center">
  <a href="https://arxiv.org/abs/123" target="_blank">📄ArXiv Paper</a>
  • <a href="https://123" target="_blank">🌩️Google Drive</a>
  • <a href="https://huggingface.co/123" target="_blank">🤗Hugging Face</a>
</p>

- [⚙️ Set up Environment](#️-set-up-environment)
- [🗺️ Navigation Task](#️-navigation-task)
- [👀 Perception Task](#-perception-task)
  - [Using the Bench to Eval](#using-the-bench-to-eval)
    - [Import Data](#import-data)
    - [Set your Model Parameters](#set-your-model-parameters)
    - [Simple Multi-views](#simple-multi-views)
    - [Multi-views with Sonar](#multi-views-with-sonar)
    - [Multi-views add Sonar Examples](#multi-views-add-sonar-examples)
  - [Collecting Image Data (Optional)](#collecting-image-data-optional)
    - [Modify Configuration File](#modify-configuration-file)
    - [Collect Camera Images Only](#collect-camera-images-only)
    - [Collect Camera and Sonar Images](#collect-camera-and-sonar-images)
- [🚩 Citation](#-citation)

# ⚙️ Set up Environment

```bash
conda create -n oceangym python=3.13.2
conda activate oceangym
pip install -r requirements.txt
```

after that, make sure holoocean is ready.

**follow this carefully:** https://byu-holoocean.github.io/holoocean-docs/v2.0.0/usage/installation.html

# 🗺️ Navigation Task

# 👀 Perception Task

## Using the Bench to Eval

### Import Data

First, you need download our data from [Google Drive](todo) or [Hugging Face](todo).

And then create a new `data` folder in the project root directory:

```bash
mkdir -p data/perception
```

Finally, put the downloaded data into the corresponding folder.

### Set your Model Parameters

Open a terminal in the root directory and set it directly, or use the script from [here](scripts\perception\eval.sh).

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
python eval\perception\mv.py \
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
python eval\perception\mv.py \
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
python eval\perception\mv.py \
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
python eval\perception\mv.py \
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
python eval\perception\mvs.py \
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
python eval\perception\mvs.py \
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
python eval\perception\mvs.py \
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
python eval\perception\mvs.py \
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
python eval\perception\mvsex.py \
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
python eval\perception\mvsex.py \
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
python eval\perception\mvsex.py \
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
python eval\perception\mvsex.py \
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

The sample configuration files can be found in `example\perception\map_config`. You need to copy this and paste it into your HoloOcean project's configuration.

### Collect Camera Images Only

This command is used to collect **camera** images only, and the parameters are as follows:

| parameter | function |
| ---| --- |
| scenario | The name of the json configuration file you want to replace. |
| task_name | Customize the name of the experiment to save the results. |
| rgbcamera | The camera directions you can choose. If select all, enter "all". |

```bash
python task\perception\init_map.py \
    --scenario without_sonar \
    --task_name "Exp_Camera_Only" \
    --rgbcamera "all"
```

### Collect Camera and Sonar Images

This command is used to collect both **camera** images and **sonar** images at same time:

```bash
python task\perception\init_map_with_sonar.py \
    --scenario with_sonar \
    --task_name "Exp_Add_Sonar" \
    --rgbcamera "FrontCamera"
```

# 🚩 Citation

If this paper or benchmark is helpful, please kindly cite as this, thanks!

```bibtex
@inproceedings{oceangym,
  title={OceanGym: A Benchmark Environment for Underwater Embodied Agents},
  ...
}
```

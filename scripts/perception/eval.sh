# Define your model parameters here

MODELS_TEMPLATE="Yours"
MODEL_NAME_OR_PATH="Yours"
API_KEY="Yours"
BASE_URL="Yours"

# Task: Multi-views

python eval\perception\mv.py \
    --exp_name Result_MV_highLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLight.json" \
    --images_dir "/data/perception/highLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

python eval\perception\mv.py \
    --exp_name Result_MV_highLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLightContext.json" \
    --images_dir "/data/perception/highLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

python eval\perception\mv.py \
    --exp_name Result_MV_lowLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLight.json" \
    --images_dir "/data/perception/lowLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

python eval\perception\mv.py \
    --exp_name Result_MV_lowLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLightContext.json" \
    --images_dir "/data/perception/lowLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

# Task: Multi-views with Sonar

python eval\perception\mvs.py \
    --exp_name Result_MVwS_highLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLight.json" \
    --images_dir "/data/perception/highLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

python eval\perception\mvs.py \
    --exp_name Result_MVwS_highLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLightContext.json" \
    --images_dir "/data/perception/highLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

python eval\perception\mvs.py \
    --exp_name Result_MVwS_lowLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLight.json" \
    --images_dir "/data/perception/lowLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

python eval\perception\mvs.py \
    --exp_name Result_MVwS_lowLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLightContext.json" \
    --images_dir "/data/perception/lowLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

# Task: Multi-views with Sonar Examples

python eval\perception\mvsex.py \
    --exp_name Result_MVwSss_highLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLight.json" \
    --images_dir "/data/perception/highLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

python eval\perception\mvsex.py \
    --exp_name Result_MVwSss_highLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/highLightContext.json" \
    --images_dir "/data/perception/highLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

python eval\perception\mvsex.py \
    --exp_name Result_MVwSss_lowLight_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLight.json" \
    --images_dir "/data/perception/lowLight" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

python eval\perception\mvsex.py \
    --exp_name Result_MVwSss_lowLightContext_00 \
    --exp_idx "all" \
    --exp_json "/data/perception/lowLightContext.json" \
    --images_dir "/data/perception/lowLightContext" \
    --model_template $MODELS_TEMPLATE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --api_key $API_KEY \
    --base_url $BASE_URL

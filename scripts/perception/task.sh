# task 01: Collect camera images only

python task\perception\init_map.py \
    --scenario without_sonar \
    --task_name "Exp_Camera_Only" \
    --rgbcamera "all"

# task 02: Collect sonar images while collecting camera images

python task\perception\init_map_with_sonar.py \
    --scenario with_sonar \
    --task_name "Exp_Add_Sonar" \
    --rgbcamera "FrontCamera"

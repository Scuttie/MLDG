#!/bin/bash
#SBATCH --job-name=LODO_MAML_GridSearch_PACS
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --partition=laal_3090
#SBATCH --time=UNLIMITED

# --- ⚙️ 1. Grid Search 설정: 여기서 탐색할 하이퍼파라미터 값들을 정의합니다 ---

# 메타 학습률(meta_lr) 후보들 (기존 값 1e-4 주변과 더 낮은 값들)
META_LRS=(1e-5 1e-6 1e-7) 

# 내부 학습률(inner_lr) 후보들 (기존 값 1e-7 주변과 더 낮은 값들)
INNER_LRS=(1e-3 1e-4 1e-5)

LAMBDA_DAS=(1.0)

# 실험할 데이터셋 목록 (테스트를 위해 PACS 하나만 남겼습니다. 필요시 추가하세요.)
# "VLCS" "TerraIncognita" "DomainNet" "OfficeHome" "PACS"
DATASETS=("PACS") 

# --- 공통 경로 설정 ---
DATA_DIR="/home/shared"
BASE_OUTPUT_DIR="./output"

# --- 🚀 2. Grid Search 실행 로직 ---

# 데이터셋에 대한 반복
for dataset in "${DATASETS[@]}"; do
    # meta_lr에 대한 반복
    for meta_lr in "${META_LRS[@]}"; do
        # inner_lr에 대한 반복
        for inner_lr in "${INNER_LRS[@]}"; do
            for lambda_da in "${LAMBDA_DAS[@]}"; do
                # ✅ 각 실험 결과를 저장할 고유한 폴더 이름 생성
                # 예: LODO_MAML_PACS_meta1e-5_inner1e-8
                OUTPUT_FOLDER_NAME="LODO_MAML_${dataset}_meta${meta_lr}_inner${inner_lr}_lambda-da${lambda_da}"

                echo "================================================="
                echo "🚀 Starting Grid Search Run:"
                echo "  - Dataset:    ${dataset}"
                echo "  - Meta LR:    ${meta_lr}"
                echo "  - Inner LR:   ${inner_lr}"
                echo "  - Lambda DA:   ${lambda_da}"
                echo "  - Output Dir: ${BASE_OUTPUT_DIR}/${OUTPUT_FOLDER_NAME}"
                echo "================================================="

                # ✅ 변수화된 하이퍼파라미터로 Python 스크립트 실행
                python -m mdlt.train \
                --dataset "${dataset}" \
                --algorithm "LODO_DA_MAML" \
                --use_meta_learning True \
                --output_folder_name "${OUTPUT_FOLDER_NAME}" \
                --data_dir "${DATA_DIR}" \
                --output_dir "${BASE_OUTPUT_DIR}" \
                --seed 0 \
                --use_boda False \
                --use_calibration False \
                --meta_lr ${meta_lr} \
                --inner_lr ${inner_lr} \
                --meta_beta 1.0 \
                --lambda_da ${lambda_da} \
                --hparams '{"temperature": 1.0, "nu": 0.5, "cross_env_gamma": 1.0, "weight_decay": 5e-4}'
                
                echo "✅ Finished run for ${OUTPUT_FOLDER_NAME}"
                echo "-------------------------------------------------"
                echo ""
            done
        done
    done
done

echo "🎉 All grid search runs are complete."

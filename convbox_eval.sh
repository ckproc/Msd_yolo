#!/bin/bash

UNDO=false

dataset="pascal_voc"
checkpoint_dir="backup/voc-dilation-aug-ems-final-448"
pascal_eval="backup/pascal_eval"
eval_dir="result"

flagfile=flags/${dataset}_eval_flags

for ((i = 20000; i <= 20000; i += 2000)); do
    checkpoint_path="${checkpoint_dir}/model.ckpt-${i}"
    result_log="${pascal_eval}/result-${i}.log"
    
    if [ -f ${result_log} ]; then
        continue
    fi
    
    while [ ! -f "${checkpoint_path}.index" ]; do
        sleep 180
    done

    # Evaluate the fine-tuned model.
    python convbox/convbox_eval.py \
        --checkpoint_path=${checkpoint_path} \
        --eval_dir=${eval_dir}/data \
        --flagfile=${flagfile}

    if ${UNDO}; then
        break
    fi

    case ${dataset} in
        "pascal_voc") matfile=convbox/utils/eval_voc/compute_mAP.m
            chmod +x ${matfile}
            matlab -nodisplay -r "run ./${matfile}; quit;" > "${result_log}"
        ;;
        "kitti") ./convbox/utils/eval_kitti/cpp/evaluate_object \
                ~/data/KITTI/data_object_label_2/training \
                data/kitti/val_idx.txt ${eval_dir} 2000
            for ap_file in ${eval_dir}/stats_*_ap.txt; do
                echo ${ap_file} >> "${result_log}"
                cat ${ap_file} >> "${result_log}"
            done
            rm ${eval_dir}/*.txt; rm -r ${eval_dir}/plot
        ;;
        *) echo "error."
        ;;
    esac
done

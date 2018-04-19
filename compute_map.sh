matfile=convbox/utils/eval_voc/compute_mAP.m
result_log="backup/voc-dilation-aug-ems-final-448/result-6000.log"
chmod +x ${matfile}
matlab -nodisplay -r "run ./${matfile}; quit;" > "${result_log}"
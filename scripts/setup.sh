data_dir="../data"
results_dir="../results"
rm -rf data
ln -s ${data_dir} data

rm -rf ckpts
ckpts_dir="${results_dir}/RE_improved_baseline/ckpts"
if [ ! -d "${ckpts_dir}" ]; then
  mkdir -p ${ckpts_dir}
fi
ln -s ${ckpts_dir} ckpts

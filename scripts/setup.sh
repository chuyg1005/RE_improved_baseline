rm -rf data
ln -s /home/data_91_d/chuyg/mix-debias/data data

rm -rf ckpts
if [ ! -d "/home/data_91_d/chuyg/mix-debias/RE_improved_baseline/ckpts" ]; then
  mkdir -p /home/data_91_d/chuyg/mix-debias/RE_improved_baseline/ckpts
fi
ln -s /home/data_91_d/chuyg/mix-debias/RE_improved_baseline/ckpts ckpts

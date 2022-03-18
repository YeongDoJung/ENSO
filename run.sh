python main.py --name oisst_lstm_rfb4_rmse --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 0  --crit mse
python main.py --name oisst_lstm_rfb4_weightedrmse --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 1 --crit weightedMSE
python main.py --name oisst_lstm_rfb4_gumbel --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 2 --crit GGELV
python main.py --name oisst_lstm_rfb4_frechet --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 3 --crit FGELV
python main.py --name oisst_trf_rfb4_rmse --data 1 --dataset oisst3 --model oisst_encoder --batch_size 16 --gpu 4 --crit mse
python main.py --name oisst_trf_rfb4_weightedrmse --data 1 --dataset oisst3 --model oisst_encoder --batch_size 16 --gpu 5 --crit weightedMSE
python main.py --name oisst_trf_rfb4_gumbel --data 1 --dataset oisst3 --model oisst_encoder --batch_size 16 --gpu 6 --crit GGELV
python main.py --name oisst_trf_rfb4_frechet --data 1 --dataset oisst3 --model oisst_encoder --batch_size 16 --gpu 7 --crit FGELV

python main.py --name oisst_sst_trf_mse --data 1 --dataset oisst3_sstonly --model oisst_encoder_sst --batch_size 10 --gpu 4 --crit mse
python main.py --name oisst_sst_trf_wm --data 1 --dataset oisst3_sstonly --model oisst_encoder_sst --batch_size 10 --gpu 5 --crit weightedmse
python main.py --name oisst_sst_trf_gumbel --data 1 --dataset oisst3_sstonly --model oisst_encoder_sst --batch_size 10 --gpu 6 --crit GGELV
python main.py --name oisst_sst_trf_frechet --data 1 --dataset oisst3_sstonly --model oisst_encoder_sst --batch_size 10 --gpu 7 --crit FGELV



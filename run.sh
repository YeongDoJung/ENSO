python main.py --name oisst_lstm_rfb4_rmse --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 0  --crit mse
python main.py --name oisst_lstm_rfb4_weightedrmse --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 1 --crit weightedMSE
python main.py --name oisst_lstm_rfb4_gumbel --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 2 --crit GGELV
python main.py --name oisst_lstm_rfb4_frechet --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 3 --crit FGELV
python main.py --name oisst_trf_rfb4_rmse --data 1 --dataset oisst3 --model oisst_encoder --batch_size 100 --gpu 4 --crit mse
python main.py --name oisst_trf_rfb4_weightedrmse --data 1 --dataset oisst3 --model oisst_encoder --batch_size 100 --gpu 5 --crit weightedMSE
python main.py --name oisst_trf_rfb4_gumbel --data 1 --dataset oisst3 --model oisst_encoder --batch_size 100 --gpu 6 --crit GGELV
python main.py --name oisst_trf_rfb4_frechet --data 1 --dataset oisst3 --model oisst_encoder --batch_size 100 --gpu 7 --crit FGELV

python main.py --name oisst_trf_edit --data 1 --dataset oisst3 --model oisst_encoder_edit --batch_size 10 --gpu 3 --crit FGELV



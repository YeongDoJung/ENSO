python main.py --name oisst_lstm_rfb4_rmse --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 0  --crit mse
python main.py --name oisst_lstm_rfb4_weightedrmse --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 1 --crit weightedMSE
python main.py --name oisst_lstm_rfb4_gumbel --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 2 --crit GGELV
python main.py --name oisst_lstm_rfb4_frechet --data 1 --dataset oisst3 --model oisst_Model_3D --batch_size 100 --gpu 3 --crit FGELV
python main.py --name oisst_trf_rfb4_rmse --data 1 --dataset oisst3 --model oisst_encoder --batch_size 16 --gpu 4 --crit mse
python main.py --name oisst_trf_rfb4_weightedrmse --data 1 --dataset oisst3 --model oisst_encoder --batch_size 16 --gpu 5 --crit weightedMSE
python main.py --name oisst_trf_rfb4_gumbel --data 1 --dataset oisst3 --model oisst_encoder --batch_size 16 --gpu 6 --crit GGELV
python main.py --name oisst_trf_rfb4_frechet --data 1 --dataset oisst3 --model oisst_encoder --batch_size 16 --gpu 7 --crit FGELV

python main.py --name sep_pvt_ --data 1 --dataset oisst3 --model sep_pvt --batch_size 10 --gpu 0 --crit mse
python main.py --name sep_pvt_wmse --data 1 --dataset oisst3 --model sep_pvt --batch_size 10 --gpu 1 --crit weightedmse
python main.py --name sep_pvt_gv --data 1 --dataset oisst3 --model sep_pvt --batch_size 10 --gpu 2 --crit GGELV

python main.py --data 1 --dataset oisst3 --batch_size 10 --gpu 3 --crit FGELV --name sep_pvt_fr --model sep_pvt

python main.py --name sc_predict_sep_pvt --data 1 --dataset oisst3 --model sep_pvt --batch_size 10 --gpu 7 --crit mse 
python main.py --name 12m_predict_sep_pvt_fr --data 1 --dataset oisst3 --model sep_pvt --batch_size 10 --gpu 4 --crit FGELV --data_targetmonth 12
python main.py --name 12m_predict_sep_pvt_gv --data 1 --dataset oisst3 --model sep_pvt --batch_size 10 --gpu 5 --crit GGELV --data_targetmonth 12

python tgt_main.py --name st_enc_mse --data 1 --dataset oisst_tgt --model sattr_enc --batch_size 20 --crit mse --gpu 3 --data_inputmonth 3 --data_targetmonth 24 
python tgt_main.py --name st_enc_gb --data 1 --dataset oisst_tgt --model sattr_enc --batch_size 20 --crit GGELV --gpu 4 --data_inputmonth 3 --data_targetmonth 24
python tgt_main.py --name st_enc_fr --data 1 --dataset oisst_tgt --model sattr_enc --batch_size 20 --crit FGELV --gpu 5 --data_inputmonth 3 --data_targetmonth 24

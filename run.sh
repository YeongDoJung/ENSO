python main.py --name oisst_lstm_rfb4_rmse --data 0 --dataset basicdataset --model oisst_Model_3D --batch_size 100 --gpu 0
python main.py --name oisst_lstm_rfb4_weightedrmse --data 0 --dataset basicdataset --model oisst_Model_3D --batch_size 100 --gpu 1
python main.py --name oisst_lstm_rfb4_gumbel --data 0 --dataset basicdataset --model oisst_Model_3D --batch_size 100 --gpu 2
python main.py --name oisst_lstm_rfb4_frechet --data 0 --dataset basicdataset --model oisst_Model_3D --batch_size 100 --gpu 3
python main.py --name oisst_trf_rfb4_rmse --data 0 --dataset basicdataset --model oisst_encoder --batch_size 100 --gpu 4
python main.py --name oisst_trf_rfb4_weightedrmse --data 0 --dataset basicdataset --model oisst_encoder --batch_size 100 --gpu 5
python main.py --name oisst_trf_rfb4_gumbel --data 0 --dataset basicdataset --model oisst_encoder --batch_size 100 --gpu 6
python main.py --name oisst_trf_rfb4_frechet --data 0 --dataset basicdataset --model oisst_encoder --batch_size 100 --gpu 7
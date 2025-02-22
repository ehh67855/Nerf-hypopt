# Nerf Hyper Parameter Optimization

##Setup on juncus

```
pip uninstall torch torchvision torchaudio -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python run_nerf.py --netdepth 10 --netwidth 512 --lrate 0.001 --N_rand 512 --config configs/lego.txt 
```
